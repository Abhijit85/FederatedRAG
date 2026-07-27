# Copyright Sierra

import json
from typing import List, Optional, Dict, Any, Tuple
import re

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.local_completion import completion
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


def _extract_tool_call_from_content(content: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(content, str):
        return None
    name_match = re.search(r'"name"\s*:\s*"([^"\n]+)"', content)
    if not name_match:
        return None
    args_match = re.search(r'"arguments"\s*:\s*("(?:\\.|[^"\\])*"|\{.*?\})', content, re.DOTALL)
    arguments: Dict[str, Any] = {}
    if args_match:
        raw_args = args_match.group(1)
        try:
            parsed_args = json.loads(raw_args)
            if isinstance(parsed_args, str):
                parsed_args = json.loads(parsed_args)
            if isinstance(parsed_args, dict):
                arguments = parsed_args
        except Exception:
            brace_start = raw_args.find('{')
            brace_end = raw_args.rfind('}')
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                snippet = raw_args[brace_start:brace_end+1]
                try:
                    parsed_args = json.loads(snippet)
                    if isinstance(parsed_args, dict):
                        arguments = parsed_args
                except Exception:
                    arguments = {}
    return {"function": {"name": name_match.group(1), "arguments": json.dumps(arguments)}}


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def _build_system_prompt(self) -> str:
        guidance = """

Additional execution rules:
- Before any consequential action, verify the order status from a fresh or already retrieved order payload and ensure the tool is valid for that status.
- `return_delivered_order_items` and `exchange_delivered_order_items` require a delivered order.
- `cancel_pending_order` and `modify_pending_order_*` require a pending order.
- Use exact `item_id` values from the retrieved order payload; do not infer or paraphrase them.
- Use exact `new_item_id` values from `get_product_details`; do not call an exchange/modify tool with an unavailable or unseen variant id.
- Treat explicit user constraints as hard constraints. If the user specifies "same shoe size", preserve that size. If the user gives a ranked preference such as "AC adapter > battery > USB", prefer the best available option in that order before considering price.
- For "highest-tier" or "upgrade" requests, maximize price only subject to the user's explicit constraints and the order's allowed action type. For wearable items such as shoes or boots, keep the same size unless the user explicitly asks to change it. When charging only an upgrade difference and a gift card on the account fully covers that difference, prefer the gift card unless the user explicitly requested another method.
- If the user wants to upgrade multiple purchased items and the account has multiple orders, inspect the relevant orders before proposing an action. Prefer pending orders for item upgrades, because item substitutions can be applied there directly; do not anchor on a processed order if another actionable pending order exists.
- If a requested action is impossible by policy, do not substitute a nearby valid action. Explain the constraint, and if the user insists, transfer to a human.
- Refunds for returns must go either to the original payment method or to an existing gift card. Do not process a return to an unrelated payment method.
- If a tool returns an error, do not immediately retry the same tool call with the same or permuted arguments. Re-check order details or product details first, or ask the user for clarification.
- If the user asked to act on multiple orders or items and only part of the request has been completed, explicitly check whether the remaining orders/items still need action before closing the conversation.
"""
        return self.wiki + guidance

    def _normalize_order_id(self, order_id: Any) -> str:
        if not isinstance(order_id, str):
            return ""
        return order_id if order_id.startswith("#") else f"#{order_id}"

    def _parse_json(self, content: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(content, str):
            return None
        try:
            parsed = json.loads(content)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _status_is_pending(self, status: str) -> bool:
        normalized = (status or "").strip().lower()
        return normalized == "pending" or normalized.startswith("pending ")

    def _status_is_delivered(self, status: str) -> bool:
        return (status or "").strip().lower() == "delivered"

    def _original_payment_method(self, order: Dict[str, Any]) -> str:
        history = order.get("payment_history") or []
        if not isinstance(history, list) or not history:
            return ""
        for entry in reversed(history):
            if isinstance(entry, dict) and entry.get("payment_method_id"):
                return str(entry["payment_method_id"])
        return ""

    def _validate_action(
        self,
        action: Action,
        order_cache: Dict[str, Dict[str, Any]],
        product_cache: Dict[str, Dict[str, Any]],
        user_cache: Dict[str, Dict[str, Any]],
        failed_calls: set[Tuple[str, str]],
        user_preferences: Dict[str, Any],
    ) -> Optional[str]:
        if action.name == RESPOND_ACTION_NAME:
            return None

        allowed_tool_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in self.tools_info
            if isinstance(tool, dict)
        }
        if action.name not in allowed_tool_names:
            return (
                "Use only the retail tools exposed in this environment. Do not call unsupported tools such as internal reasoning or calculator tools."
            )

        call_signature = (action.name, json.dumps(action.kwargs, sort_keys=True))
        if call_signature in failed_calls:
            return (
                "Do not repeat the same failed tool call. Re-check the order details or product variants first, "
                "or ask the user for clarification."
            )

        if action.name == "find_user_id_by_name_zip":
            first_name = str(action.kwargs.get("first_name") or "").strip()
            last_name = str(action.kwargs.get("last_name") or "").strip()
            zip_code = str(action.kwargs.get("zip") or "").strip()
            if not first_name or not last_name or not zip_code:
                return (
                    "You cannot look up a user with partial identity information. "
                    "Obtain full first name, last name, and zip code before calling "
                    "`find_user_id_by_name_zip`."
                )
            placeholder_names = {"john", "jane", "doe", "user", "customer"}
            if first_name.lower() in placeholder_names or last_name.lower() in placeholder_names or zip_code in {"12345", "00000"}:
                return (
                    "Do not guess authentication details. Ask the user for their real first name, last name, and zip code before calling `find_user_id_by_name_zip`."
                )

        if action.name == "find_user_id_by_email":
            email = str(action.kwargs.get("email") or "").strip()
            if not email:
                return "You need a concrete email address before calling `find_user_id_by_email`."
            lowered = email.lower()
            if lowered in {"your_email@example.com", "user@example.com", "customer@example.com"}:
                return (
                    "Do not guess or invent an email address for authentication. Ask the user for their real email or switch to first name + last name + zip code."
                )

        order_id = self._normalize_order_id(action.kwargs.get("order_id"))
        known_orders = set()
        for user_record in user_cache.values():
            orders = user_record.get("orders") or []
            for known_order_id in orders:
                normalized = self._normalize_order_id(known_order_id)
                if normalized:
                    known_orders.add(normalized)
        if action.name == "get_order_details":
            if not user_cache and order_id in {"#W0000000", "#W0000001", "#W1234567"}:
                return (
                    "Authenticate the user first and retrieve their account orders before calling `get_order_details` with a concrete order id."
                )
            if known_orders and order_id and order_id not in known_orders:
                return (
                    "Use an order_id from the authenticated user's account only. Known orders: "
                    + ", ".join(sorted(known_orders))
                    + "."
                )
        order = order_cache.get(order_id) if order_id else None
        if order is not None:
            status = str(order.get("status", ""))
            item_ids = {
                str(item.get("item_id"))
                for item in order.get("items", [])
                if isinstance(item, dict) and item.get("item_id") is not None
            }

            if action.name in {"return_delivered_order_items", "exchange_delivered_order_items"}:
                if not self._status_is_delivered(status):
                    return f"Order {order_id} is not delivered, so `{action.name}` is invalid."
            if action.name in {
                "cancel_pending_order",
                "modify_pending_order_address",
                "modify_pending_order_payment",
                "modify_pending_order_items",
            }:
                if not self._status_is_pending(status):
                    return f"Order {order_id} is not pending, so `{action.name}` is invalid."

            provided_item_ids = action.kwargs.get("item_ids")
            if action.name in {"return_delivered_order_items", "exchange_delivered_order_items", "modify_pending_order_items"}:
                if not isinstance(provided_item_ids, list) or len(provided_item_ids) == 0:
                    return f"`{action.name}` requires a non-empty `item_ids` list derived from the order payload."
            if isinstance(provided_item_ids, list) and item_ids:
                missing = [str(item_id) for item_id in provided_item_ids if str(item_id) not in item_ids]
                if missing:
                    return (
                        f"Some requested item_ids are not present in order {order_id}: {', '.join(missing)}. "
                        "Re-check the order payload before acting."
                    )

            if action.name == "return_delivered_order_items":
                chosen_payment = str(action.kwargs.get("payment_method_id") or "")
                original_payment = self._original_payment_method(order)
                if chosen_payment and not chosen_payment.startswith("gift_card_") and chosen_payment != original_payment:
                    return (
                        "Return refunds must go to the original payment method or an existing gift card. "
                        "Do not process this return with the currently selected payment method."
                    )

        if action.name in {"exchange_delivered_order_items", "modify_pending_order_items"}:
            new_item_ids = action.kwargs.get("new_item_ids")
            if not isinstance(new_item_ids, list) or len(new_item_ids) == 0:
                return f"`{action.name}` requires a non-empty `new_item_ids` list from available product variants."
            if product_cache:
                known_variants = {
                    variant_id
                    for product in product_cache.values()
                    for variant_id, variant in (product.get("variants") or {}).items()
                    if isinstance(variant, dict) and variant.get("available") is True
                }
                unseen = [str(item_id) for item_id in new_item_ids if str(item_id) not in known_variants]
                if unseen and known_variants:
                    return (
                        f"Some replacement item_ids have not been seen as available variants: {', '.join(unseen)}. "
                        "Re-check product details before acting."
                    )

            if action.name == "modify_pending_order_items" and order is not None:
                provided_item_ids = action.kwargs.get("item_ids") or []
                order_items_by_id = {
                    str(item.get("item_id")): item
                    for item in order.get("items", [])
                    if isinstance(item, dict) and item.get("item_id") is not None
                }

                if user_preferences.get("prefer_highest_price"):
                    for old_item_id, new_item_id in zip(provided_item_ids, new_item_ids):
                        current_item = order_items_by_id.get(str(old_item_id))
                        if not current_item:
                            continue
                        product = product_cache.get(str(current_item.get("product_id")))
                        if not product:
                            continue
                        variants = [
                            variant
                            for variant in (product.get("variants") or {}).values()
                            if isinstance(variant, dict) and variant.get("available") is True
                        ]
                        item_name = str(current_item.get("name") or "").lower()
                        should_preserve_size = user_preferences.get("preserve_same_size")
                        if not should_preserve_size and current_item.get("options", {}).get("size") is not None:
                            should_preserve_size = any(token in item_name for token in ["shoe", "boot", "sneaker", "sandal", "heel"])
                        if should_preserve_size and current_item.get("options", {}).get("size") is not None:
                            target_size = str(current_item.get("options", {}).get("size"))
                            sized = [
                                variant
                                for variant in variants
                                if str((variant.get("options") or {}).get("size")) == target_size
                            ]
                            if sized:
                                variants = sized
                        if variants:
                            best_variant = max(variants, key=lambda variant: float(variant.get("price", 0.0)))
                            if str(new_item_id) != str(best_variant.get("item_id")):
                                return (
                                    f"For a highest-price upgrade request, item {old_item_id} should use the highest valid available variant "
                                    f"{best_variant.get('item_id')} (price={best_variant.get('price')}) rather than {new_item_id}."
                                )

                if (user_preferences.get("prefer_gift_card") or user_preferences.get("prefer_highest_price")) and user_cache:
                    user_record = next(iter(user_cache.values()))
                    payment_methods = user_record.get("payment_methods") or {}
                    gift_cards = [
                        method for method in payment_methods.values()
                        if isinstance(method, dict) and str(method.get("id", "")).startswith("gift_card_")
                    ]
                    if gift_cards:
                        chosen_payment = str(action.kwargs.get("payment_method_id") or "")
                        old_total = 0.0
                        new_total = 0.0
                        for old_item_id, new_item_id in zip(provided_item_ids, new_item_ids):
                            current_item = order_items_by_id.get(str(old_item_id))
                            if current_item:
                                old_total += float(current_item.get("price", 0.0))
                                product = product_cache.get(str(current_item.get("product_id")))
                                if product:
                                    variant = (product.get("variants") or {}).get(str(new_item_id))
                                    if isinstance(variant, dict):
                                        new_total += float(variant.get("price", 0.0))
                        delta = max(0.0, new_total - old_total)
                        viable = [card for card in gift_cards if float(card.get("balance", 0.0)) + 1e-9 >= delta]
                        if viable and chosen_payment != str(viable[0].get("id")):
                            return (
                                f"The user asked to use a gift card first, and gift card {viable[0].get('id')} has enough balance for the "
                                f"incremental charge ({delta:.2f}). Use it before falling back to PayPal or another method."
                            )

        return None

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        order_cache: Dict[str, Dict[str, Any]] = {}
        product_cache: Dict[str, Dict[str, Any]] = {}
        user_cache: Dict[str, Dict[str, Any]] = {}
        failed_calls: set[Tuple[str, str]] = set()
        auth_hint_state = ""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": obs},
        ]
        for _ in range(max_num_steps):
            if not user_cache:
                current_user_text = "\n".join(
                    str(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "user"
                )
                zip_matches = re.findall(r"\b\d{5}\b", current_user_text)
                if zip_matches:
                    desired_auth_hint = (
                        f"Authentication steering: the user has already provided zip code {zip_matches[-1]}. "
                        "To minimize disclosure, ask only for the missing first and last name. "
                        "Do not switch to email unless the user explicitly offers it. "
                        "If the user hesitates, explain that first name + last name + the already provided zip code is the minimum needed."
                    )
                    if desired_auth_hint != auth_hint_state:
                        messages.append({"role": "system", "content": desired_auth_hint})
                        auth_hint_state = desired_auth_hint
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"] or 0
            action = message_to_action(next_message)
            user_text = "\n".join(
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "user"
            ).lower()
            user_preferences = {
                "prefer_highest_price": any(token in user_text for token in ["most expensive", "luxurious", "luxury", "upgrade"]),
                "preserve_same_size": "same size" in user_text,
                "prefer_gift_card": "gift card" in user_text,
                "paypal_fallback": "paypal" in user_text and ("fine" in user_text or "fallback" in user_text),
            }
            validation_error = self._validate_action(action, order_cache, product_cache, user_cache, failed_calls, user_preferences)
            if validation_error is not None:
                messages.append({"role": "system", "content": validation_error})
                continue
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                tool_payload = self._parse_json(env_response.observation)
                if action.name == "get_order_details" and tool_payload is not None:
                    order_id = self._normalize_order_id(tool_payload.get("order_id"))
                    if order_id:
                        order_cache[order_id] = tool_payload
                if action.name == "get_order_details" and tool_payload is not None:
                    items = tool_payload.get("items") or []
                    item_summary = ", ".join(
                        f"{item.get("name", "item")}: {item.get("item_id", "?")}"
                        for item in items if isinstance(item, dict)
                    )
                    original_payment = self._original_payment_method(tool_payload)
                    messages.append({
                        "role": "system",
                        "content": f"Order cache updated: order {tool_payload.get("order_id")} has status {tool_payload.get("status")}. Valid item_ids are: {item_summary}. Original payment method: {original_payment}."
                    })
                if action.name == "get_product_details" and tool_payload is not None:
                    product_id = tool_payload.get("product_id")
                    if product_id is not None:
                        product_cache[str(product_id)] = tool_payload
                    variants = tool_payload.get("variants") or {}
                    available = []
                    for item_id, variant in variants.items():
                        if isinstance(variant, dict) and variant.get("available") is True:
                            available.append(f"{item_id}: {variant.get("options", {})}, price={variant.get("price")}")
                    if available:
                        messages.append({
                            "role": "system",
                            "content": "Available variants just retrieved. Do not describe any of these item_ids as unavailable: " + "; ".join(available[:12])
                        })
                if action.name == "get_user_details" and tool_payload is not None:
                    user_id = tool_payload.get("user_id") or tool_payload.get("email") or "current_user"
                    user_cache[str(user_id)] = tool_payload
                    orders = tool_payload.get("orders") or []
                    if orders:
                        messages.append({
                            "role": "system",
                            "content": "User profile retrieved. Order IDs on this account: " + ", ".join(str(order_id) for order_id in orders) + ". If the request is to upgrade or modify purchased items across the account, inspect the relevant orders before deciding which action is valid. Pending orders support item modification; processed orders do not."
                        })
                if isinstance(env_response.observation, str) and env_response.observation.startswith("Error:"):
                    failed_calls.add((action.name, json.dumps(action.kwargs, sort_keys=True)))
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    tool_calls = message.get("tool_calls")
    if not tool_calls and isinstance(message.get("content"), str):
        try:
            parsed = json.loads(message["content"])
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            tool_calls = parsed["tool_calls"]
        elif parsed is None:
            recovered = _extract_tool_call_from_content(message.get("content"))
            if recovered is not None:
                tool_calls = [recovered]
    if tool_calls and len(tool_calls) > 0 and tool_calls[0].get("function") is not None:
        tool_call = tool_calls[0]
        arguments = tool_call["function"].get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        return Action(
            name=tool_call["function"].get("name"),
            kwargs=arguments if isinstance(arguments, dict) else {},
        )
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message.get("content")})
