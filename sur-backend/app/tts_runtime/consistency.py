"""One voice per scene.

If a scene's lines came from more than one model, the whole scene is
re-rendered with the highest-ranked model (chain order) that passes every
line in it; renders it already has are reused from the cache. If no model
passes every line, the scene stays mixed and that is recorded -- a mixed
scene is better than a silent line.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from app.tts_runtime.failover import Context, LineResult, try_model

logger = logging.getLogger(__name__)


def scenes_to_unify(results: dict[int, tuple[dict, LineResult]]) -> dict[str, list[int]]:
    by_scene: dict[str, list[int]] = defaultdict(list)
    for idx, (line, _res) in results.items():
        if line.get("scene") is not None:
            by_scene[str(line["scene"])].append(idx)
    mixed = {}
    for scene, idxs in by_scene.items():
        models = {results[i][1].chosen.model for i in idxs if results[i][1].chosen is not None}
        if len(models) > 1:
            mixed[scene] = sorted(idxs)
    return mixed


def unify(ctx: Context, results: dict[int, tuple[dict, LineResult]]) -> dict[str, str]:
    """Re-render mixed scenes. Mutates `results`; returns scene -> outcome."""
    outcomes = {}
    for scene, idxs in scenes_to_unify(results).items():
        lines = [results[i][0] for i in idxs]
        langs = {ln["lang"] for ln in lines}
        used = sorted({results[i][1].chosen.model for i in idxs if results[i][1].chosen})
        chain = ctx.cfg.chain_for(next(iter(langs))) if len(langs) == 1 else ()
        winner = None
        for model in chain:
            ok, _why = ctx.available(model)
            if not ok or not ctx.breaker.allows(model, lines[0]["lang"]):
                continue
            if not all(adapter_supports(model, ln["lang"]) for ln in lines):
                continue
            new = {}
            for ln in lines:
                tries = try_model(ctx, ln, model, set_state=False)
                if not (tries and tries[-1].passed):
                    break
                new[ln["idx"]] = tries[-1]
            else:
                winner = model
                for i, t in new.items():
                    line, old = results[i]
                    results[i] = (line, LineResult("DONE", t, old.tries + [t], "", old.unverified))
                break
        if winner:
            outcomes[scene] = f"re-rendered with {winner}"
            ctx.store.event("scene_rerender", ctx.job_id, model=winner,
                            detail={"scene": scene, "lines": idxs, "was": used})
            logger.warning("scene %s mixed %s; re-rendered every line with %s", scene, used, winner)
        else:
            outcomes[scene] = "kept mixed: no model passes every line"
            ctx.store.event("scene_mixed", ctx.job_id, detail={"scene": scene, "lines": idxs, "models": used})
            logger.warning("scene %s stays mixed %s: no model passes all %d lines", scene, used, len(idxs))
    return outcomes


def adapter_supports(model: str, lang: str) -> bool:
    from app.tts_runtime.adapters import adapter_class

    return adapter_class(model).supports(lang)
