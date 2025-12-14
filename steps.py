"""
Step functions for the Idea Unpacker flow.
"""

import asyncio
from schemas import (
    UserInput, Idea, ScoredIdea, FormatSpec, Draft, 
    Evaluation, OutputFormat
)
from llm_clients import (
    call_claude, call_gpt, call_gemini, call_deepseek, 
    parse_json_response
)
import config


# Step 1: User Input (handled in main.py via CLI)


async def step2_generate_ideas(user_input: UserInput) -> list[Idea]:
    """
    Step 2: Grounded Idea Generation via Claude.
    Returns 4-5 ideas, at least 3 from named authors.
    """
    prompt = f"""Given this topic and intent, generate 4-5 underexplored angles.

Topic: {user_input.topic}
Intent: {user_input.intent}

Requirements:
- At least 3 ideas must reference specific authors/thinkers who have written about related concepts
- 1-2 ideas can be your own synthesis (mark as model-generated)
- Focus on angles that are NOT mainstream

Return JSON array:
[{{
    "name": "short angle name",
    "description": "one sentence",
    "why_underexplored": "one sentence", 
    "source": "Author Name" or "model-generated",
    "is_model_generated": true/false
}}]

Return ONLY valid JSON, no other text."""

    response = await call_claude(prompt)
    ideas_data = parse_json_response(response)
    return [Idea(**idea) for idea in ideas_data]


async def step3_dual_scoring(ideas: list[Idea]) -> list[ScoredIdea]:
    """
    Step 3: Parallel scoring by two models (GPT + DeepSeek).
    Flags high-divergence ideas.
    """
    ideas_text = "\n".join([
        f"{i+1}. {idea.name}: {idea.description} (Source: {idea.source})"
        for i, idea in enumerate(ideas)
    ])
    
    scoring_prompt = f"""Score each idea for ORIGINALITY (1-10).
High scores = genuinely novel, underexplored, non-obvious.
Low scores = well-trodden, obvious, mainstream.

Ideas:
{ideas_text}

Return JSON array:
[{{"idea_index": 0, "score": 7.5, "rationale": "one sentence"}}]

Return ONLY valid JSON."""

    # Parallel calls
    score_1_task = call_gpt(scoring_prompt)
    score_2_task = call_deepseek(scoring_prompt)
    
    response_1, response_2 = await asyncio.gather(score_1_task, score_2_task)
    
    scores_1 = {s["idea_index"]: s for s in parse_json_response(response_1)}
    scores_2 = {s["idea_index"]: s for s in parse_json_response(response_2)}
    
    scored_ideas = []
    for i, idea in enumerate(ideas):
        s1 = scores_1.get(i, {"score": 5, "rationale": "No score"})
        s2 = scores_2.get(i, {"score": 5, "rationale": "No score"})
        delta = abs(s1["score"] - s2["score"])
        combined = (s1["score"] + s2["score"]) / 2
        
        scored_ideas.append(ScoredIdea(
            idea=idea,
            score_1=s1["score"],
            score_2=s2["score"],
            rationale_1=s1["rationale"],
            rationale_2=s2["rationale"],
            score_delta=delta,
            combined_score=combined
        ))
    
    return scored_ideas


def step3b_select_top_idea(scored_ideas: list[ScoredIdea]) -> ScoredIdea:
    """Select highest combined score. Flag divergence."""
    sorted_ideas = sorted(scored_ideas, key=lambda x: x.combined_score, reverse=True)
    top = sorted_ideas[0]
    
    if top.score_delta > config.SCORE_DIVERGENCE_THRESHOLD:
        print(f"⚡ High divergence detected (delta={top.score_delta:.1f}) — contested territory")
    
    return top


async def step5_format_and_criteria(
    selected: ScoredIdea, 
    user_input: UserInput
) -> FormatSpec:
    """
    Step 5: DeepSeek proposes format, criteria, and minimum bar.
    """
    prompt = f"""Given this idea and user intent, design the output format.

Idea: {selected.idea.name}
Description: {selected.idea.description}
User topic: {user_input.topic}
User intent: {user_input.intent}

Requirements:
- Choose format that EMBODIES the idea (not just describes it)
- Formats: poem, quotes, micro_essay, aphorisms, dialogue
- Define exactly 3 evaluation criteria including "surprise_density" (insight per sentence)
- Set minimum_bar (1-10) based on topic complexity

Return JSON:
{{
    "format_type": "micro_essay",
    "rationale": "why this format",
    "criteria": ["criterion_1", "criterion_2", "surprise_density"],
    "minimum_bar": 6.5
}}

Return ONLY valid JSON."""

    response = await call_deepseek(prompt)
    data = parse_json_response(response)
    return FormatSpec(**data)


async def step6_articulate(
    selected: ScoredIdea, 
    format_spec: FormatSpec
) -> Draft:
    """
    Step 6: Claude creates first draft within word limit.
    """
    prompt = f"""Create a {format_spec.format_type.value} that embodies this idea.

Idea: {selected.idea.name}
Description: {selected.idea.description}
Why underexplored: {selected.idea.why_underexplored}

HARD CONSTRAINT: Maximum {config.WORD_LIMIT} words for the main content.

Format requirements:
{format_spec.rationale}

Return JSON:
{{
    "content": "your {format_spec.format_type.value} here",
    "explainer": "2 sentences max explaining the core insight"
}}

Return ONLY valid JSON."""

    response = await call_claude(prompt)
    data = parse_json_response(response)
    word_count = len(data["content"].split())
    
    return Draft(
        content=data["content"],
        explainer=data["explainer"],
        word_count=word_count,
        version=1
    )


async def step7_evaluate(
    draft: Draft, 
    format_spec: FormatSpec,
    previous_scores: list[float] = None
) -> Evaluation:
    """
    Step 7: DeepSeek evaluates against criteria.
    Detects plateau if improvement stalls.
    """
    criteria_text = "\n".join([f"- {c}" for c in format_spec.criteria])
    
    prompt = f"""Evaluate this draft against the criteria.

Draft:
{draft.content}

Explainer:
{draft.explainer}

Criteria (score each 1-10):
{criteria_text}

Requirements:
- Be harsh but fair
- Feedback must be specific and actionable
- Maximum 3 feedback points

Return JSON:
{{
    "scores": {{"criterion_name": 7.5}},
    "total_score": 7.0,
    "feedback": ["specific improvement 1", "specific improvement 2"]
}}

Return ONLY valid JSON."""

    response = await call_deepseek(prompt)
    data = parse_json_response(response)
    
    # Detect plateau
    plateau = False
    if previous_scores and len(previous_scores) >= 2:
        recent_improvement = data["total_score"] - previous_scores[-1]
        prior_improvement = previous_scores[-1] - previous_scores[-2] if len(previous_scores) > 1 else 1
        if recent_improvement < config.PLATEAU_THRESHOLD and prior_improvement < config.PLATEAU_THRESHOLD:
            plateau = True
    
    return Evaluation(
        scores=data["scores"],
        total_score=data["total_score"],
        feedback=data["feedback"][:3],
        plateau_detected=plateau
    )


async def step7b_refine(
    draft: Draft, 
    evaluation: Evaluation, 
    format_spec: FormatSpec,
    selected: ScoredIdea
) -> Draft:
    """
    Step 7 (refinement): Claude incorporates feedback.
    """
    feedback_text = "\n".join([f"- {f}" for f in evaluation.feedback])
    
    prompt = f"""Improve this draft based on feedback.

Current draft:
{draft.content}

Feedback to address:
{feedback_text}

Current score: {evaluation.total_score}
Target: {format_spec.minimum_bar}

HARD CONSTRAINT: Maximum {config.WORD_LIMIT} words.

Original idea for reference:
{selected.idea.name}: {selected.idea.description}

Return JSON:
{{
    "content": "improved {format_spec.format_type.value}",
    "explainer": "2 sentences max"
}}

Return ONLY valid JSON."""

    response = await call_claude(prompt)
    data = parse_json_response(response)
    
    return Draft(
        content=data["content"],
        explainer=data["explainer"],
        word_count=len(data["content"].split()),
        version=draft.version + 1
    )


async def step8_failure_analysis(
    drafts: list[Draft],
    evaluations: list[Evaluation],
    format_spec: FormatSpec,
    selected: ScoredIdea
) -> str:
    """
    Step 8 (failure path): DeepSeek diagnoses why bar wasn't met.
    """
    history = "\n".join([
        f"V{d.version}: score={e.total_score}" 
        for d, e in zip(drafts, evaluations)
    ])
    
    prompt = f"""Analyze why this flow failed to meet the quality bar.

Idea: {selected.idea.name}
Format: {format_spec.format_type.value}
Minimum bar: {format_spec.minimum_bar}

Score history:
{history}

Final draft:
{drafts[-1].content}

Diagnose in 3 sentences max:
- Was the initial idea weak?
- Was the format wrong?
- Was execution the problem?
- Was the bar unrealistic for this topic?

Return plain text, no JSON."""

    return await call_deepseek(prompt)
