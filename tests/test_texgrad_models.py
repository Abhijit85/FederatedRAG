from synapse.training.texgrad_models import CitationAligner, EntailmentScorer


def test_entailment_scorer_fallback():
    scorer = EntailmentScorer(model_name="invalid-model-name")
    score = scorer.score("answer text", ["context text"])
    assert 0.0 <= score <= 1.0


def test_citation_aligner_fallback():
    aligner = CitationAligner(model_name="invalid-model-name")
    coverage = aligner.coverage("answer text", ["context text"])
    assert 0.0 <= coverage <= 1.0
