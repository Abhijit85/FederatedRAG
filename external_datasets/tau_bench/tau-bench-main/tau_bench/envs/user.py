# Copyright Sierra

import abc
import enum
import re

from typing import Optional, List, Dict, Any, Union

from tau_bench.local_completion import completion


class BaseUserSimulationEnv(abc.ABC):
    metadata = {}

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


def _extract_instruction_facts(instruction: Optional[str]) -> Dict[str, List[str]]:
    if not instruction:
        return {"emails": [], "zips": [], "order_ids": [], "user_ids": [], "itemish_ids": []}
    order_ids = []
    for oid in re.findall(r"#?W\d{7}", instruction):
        order_ids.append(oid if oid.startswith("#") else f"#{oid}")
    itemish = re.findall(r"\b(?:[A-Z]{2,}\d{4,}|\d{8,10}|(?:gift|credit|paypal)_card_\d+|credit_card_\d+|gift_card_\d+|paypal_\d+)\b", instruction)
    return {
        "emails": re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", instruction),
        "zips": re.findall(r"\b\d{5}\b", instruction),
        "order_ids": order_ids,
        "user_ids": re.findall(r"\b[a-z]+_[a-z]+_\d{3,}\b", instruction),
        "itemish_ids": itemish,
    }


def _sanitize_user_response(instruction: Optional[str], agent_message: str, response: str) -> str:
    facts = _extract_instruction_facts(instruction)
    response = response.strip()
    lower_agent = agent_message.lower()
    if not response or response == "###STOP###":
        return response

    unknown_email = [x for x in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", response) if x not in facts["emails"]]
    if unknown_email:
        if facts["emails"]:
            return f"The email linked to my account is {facts['emails'][0]}."
        return "I do not remember the email on the account, but I can provide my name and zip code instead."

    response_zips = re.findall(r"\b\d{5}\b", response)
    if response_zips and any(z not in facts["zips"] for z in response_zips):
        if facts["zips"]:
            return f"My zip code is {facts['zips'][0]}."
        return "I do not know the zip code from memory."

    response_order_ids = []
    for oid in re.findall(r"#?W\d{7}", response):
        response_order_ids.append(oid if oid.startswith("#") else f"#{oid}")
    if response_order_ids and any(oid not in facts["order_ids"] for oid in response_order_ids):
        if facts["order_ids"]:
            return f"The order ID is {facts['order_ids'][0]}."
        return "I do not remember the exact order ID. Please look it up from my account details."

    response_itemish = re.findall(r"\b(?:[A-Z]{2,}\d{4,}|\d{8,10}|credit_card_\d+|gift_card_\d+|paypal_\d+)\b", response)
    novel_itemish = [x for x in response_itemish if x not in facts["itemish_ids"] and x not in facts["order_ids"]]
    if novel_itemish:
        if "item id" in lower_agent or "item ids" in lower_agent:
            return "I do not know the exact item IDs. Please look them up from the order details."
        if "payment" in lower_agent or "card" in lower_agent or "gift card" in lower_agent:
            return "I do not know the exact payment method ID. Please look it up from my account or order details."
        return "I do not know the exact identifier. Please look it up using the available tools."

    if ("payment method" in lower_agent or "gift card" in lower_agent or "credit card" in lower_agent) and not facts["itemish_ids"]:
        return "Please use the original payment method on file for any required price difference."

    return response


def _instruction_fact_guard(instruction: Optional[str]) -> str:
    if not instruction:
        return ""
    facts: List[str] = []
    names = re.findall(r"You are ([A-Z][a-z]+(?: [A-Z][a-z]+)+)", instruction)
    if names:
        facts.append(f"Known customer name: {names[0]}.")
    user_ids = re.findall(r"\b[a-z]+_[a-z]+_\d{3,}\b", instruction)
    if user_ids:
        facts.append(f"Known user id: {user_ids[0]}.")
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", instruction)
    if emails:
        facts.append(f"Known email: {emails[0]}.")
    order_ids = re.findall(r"#?W\d{7}", instruction)
    if order_ids:
        normalized = order_ids[0] if order_ids[0].startswith("#") else f"#{order_ids[0]}"
        facts.append(f"Known order id: {normalized}.")
    zips = re.findall(r"\b\d{5}\b", instruction)
    if zips:
        facts.append(f"Known zip code: {zips[0]}.")
    fact_lines = "\n".join(f"- {fact}" for fact in facts) if facts else "- No explicit identity facts were provided in the instruction."
    return (
        "\nHard facts from the instruction:\n"
        + fact_lines
        + "\nUnknown facts must stay unknown. If the instruction does not explicitly give an email, zip code, item id, product id, payment method id, tracking number, or delivery date, do not invent one. Say you do not know and ask the agent to look it up with tools if needed.\n"
    )


class HumanUserSimulationEnv(BaseUserSimulationEnv):
    def reset(self, instruction: str) -> str:
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")

    def get_total_cost(self) -> float:
        return 0


class LLMUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.provider = provider
        self.total_cost = 0.0
        self.current_instruction: Optional[str] = None
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        res = completion(
            model=self.model, custom_llm_provider=self.provider, messages=messages
        )
        message = res.choices[0].message
        content = _sanitize_user_response(
            self.current_instruction,
            str(messages[-1].get("content", "")) if messages else "",
            str(message.content),
        )
        message_dict = message.model_dump()
        message_dict["content"] = content
        self.messages.append(message_dict)
        self.total_cost = res._hidden_params["response_cost"]
        return content

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (("\n\nInstruction: " + instruction + "\n") if instruction is not None else "")
        fact_guard = _instruction_fact_guard(instruction)
        return f"""You are a user interacting with an agent.{instruction_display}{fact_guard}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

    def reset(self, instruction: Optional[str] = None) -> str:
        self.current_instruction = instruction
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class ReactUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__(model=model, provider=provider)
        self.reset()

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (("\n\nInstruction: " + instruction + "\n") if instruction is not None else "")
        fact_guard = _instruction_fact_guard(instruction)
        return f"""You are a user interacting with an agent.{instruction_display}{fact_guard}
Rules:
- First, generate a Thought about what to do next (this message will not be sent to the agent).
- Then, generate a one line User Response to simulate the user's message (this message will be sent to the agent).
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as the User Response without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.

Format:

Thought:
<the thought>

User Response:
<the user response (this will be parsed and sent to the agent)>"""

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        res = completion(
            model=self.model, custom_llm_provider=self.provider, messages=messages
        )
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params["response_cost"]
        return self.parse_response(message.content)

    def reset(self, instruction: Optional[str] = None) -> str:
        self.current_instruction = instruction
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def parse_response(self, response: str) -> str:
        if "###STOP###" in response:
            return "###STOP###"
        elif "Thought:" in response:
            _, user_response = response.split("Thought:")
            return user_response.strip()
        elif "User Response:" in response:
            _, user_response = response.split("User Response:")
            return user_response.strip()
        else:
            raise ValueError(f"Invalid response format: {response}")

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class VerifyUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 3) -> None:
        self.model = model
        self.provider = provider
        self.max_attempts = max_attempts
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        attempts = 0
        cur_message = None
        while attempts < self.max_attempts:
            res = completion(
                model=self.model, custom_llm_provider=self.provider, messages=messages
            )
            cur_message = res.choices[0].message
            self.total_cost = res._hidden_params["response_cost"]
            if verify(self.model, self.provider, cur_message, messages):
                self.messages.append(cur_message.model_dump())
                return cur_message.content
            attempts += 1
        assert cur_message is not None
        return cur_message.content

    def reset(self, instruction: Optional[str] = None) -> str:
        self.current_instruction = instruction
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


def map_role_label(role: str) -> str:
    if role == "user":
        return "Customer"
    elif role == "assistant":
        return "Agent"
    else:
        return role.capitalize()


def verify(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> bool:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a Customer and an Agent. The Customer has generated a Response, and you need to verify if it is satisfactory (true) or not (false).
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

-----

Classification:"""
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    return "true" in res.choices[0].message.content.lower()


def reflect(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> str:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a (simulated) Customer and an Agent. The Customer generated a Response that was marked as unsatisfactory by you.
You need to generate a Reflection on what went wrong in the conversation, and propose a new Response that should fix the issues.
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

# Format:

Reflection:
<the reflection>

Response:
<the response (this will be parsed and sent to the agent)>"""
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    _, response = res.choices[0].message.content.split("Response:")
    return response.strip()


class ReflectionUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 2) -> None:
        self.model = model
        self.provider = provider
        self.max_attempts = max_attempts
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        cur_messages = messages.copy()
        initial_response = super().generate_next_message(cur_messages)
        if verify(self.model, self.provider, initial_response, cur_messages):
            return initial_response
        attempts = 1
        while attempts < self.max_attempts:
            new_message = reflect(
                self.model, self.provider, initial_response, cur_messages
            )
            cur_messages.append({"role": "user", "content": new_message})
            new_response = super().generate_next_message(cur_messages)
            if verify(self.model, self.provider, new_response, cur_messages):
                return new_response
            attempts += 1
        return initial_response

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class UserStrategy(enum.Enum):
    HUMAN = "human"
    LLM = "llm"
    REACT = "react"
    VERIFY = "verify"
    REFLECTION = "reflection"


def load_user(
    user_strategy: Union[str, UserStrategy],
    model: Optional[str] = "gpt-4o",
    provider: Optional[str] = None,
) -> BaseUserSimulationEnv:
    if isinstance(user_strategy, str):
        user_strategy = UserStrategy(user_strategy)
    if user_strategy == UserStrategy.HUMAN:
        return HumanUserSimulationEnv()
    elif user_strategy == UserStrategy.LLM:
        if model is None:
            raise ValueError("LLM user strategy requires a model")
        if provider is None:
            raise ValueError("LLM user strategy requires a model provider")
        return LLMUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REACT:
        if model is None:
            raise ValueError("React user strategy requires a model")
        if provider is None:
            raise ValueError("React user strategy requires a model provider")
        return ReactUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.VERIFY:
        if model is None:
            raise ValueError("Verify user strategy requires a model")
        if provider is None:
            raise ValueError("Verify user strategy requires a model provider")
        return VerifyUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REFLECTION:
        if model is None:
            raise ValueError("Reflection user strategy requires a model")
        if provider is None:
            raise ValueError("Reflection user strategy requires a model provider")
        return ReflectionUserSimulationEnv(model=model, provider=provider)
    raise ValueError(f"Unknown user strategy {user_strategy}")
