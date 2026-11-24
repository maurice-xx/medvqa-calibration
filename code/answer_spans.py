"""答案文本和生成 token 概率使用同一跨度；不加载模型。"""
from __future__ import annotations

import math
import re

SPAN_VERSION = "final-answer-v2"


def base_format(fmt):
    if not isinstance(fmt, str) or re.fullmatch(r"[ABC](?:[23])?", fmt) is None:
        raise ValueError(f"未知回答格式: {fmt!r}")
    return fmt[0]


def text_span(raw, fmt):
    """最后一个 Answer: 后至 Confidence: 起点，返回精确字符区间。"""
    base = base_format(fmt)
    markers = list(re.finditer(r"Answer:", raw)) if base != "A" else []
    missing = base != "A" and not markers
    start = markers[-1].end() if markers else 0
    end = len(raw)
    confidence_start = None
    if base == "C":
        marker = re.search(r"Confidence:", raw[start:])
        if marker:
            confidence_start = start + marker.start()
            end = confidence_start
    while start < end and (raw[start].isspace() or raw[start] == '"'):
        start += 1
    while end > start and (raw[end - 1].isspace() or raw[end - 1] == '"'):
        end -= 1
    return {
        "ans_raw": raw[start:end],
        "span_char_start": start,
        "span_char_end": end,
        "answer_marker_count": len(markers),
        "marker_missing_fallback": missing,
        "span_extraction": "missing_marker_full_text" if missing else (
            "direct_answer" if base == "A" else "last_answer_marker"),
        "confidence_marker_start": confidence_start,
        "span_version": SPAN_VERSION,
    }


def extract_answer(raw, fmt):
    return text_span(raw, fmt)["ans_raw"]


def extract_verbalized(raw, fmt="C"):
    if base_format(fmt) != "C":
        return None
    meta = text_span(raw, fmt)
    start = meta["confidence_marker_start"]
    if start is None:
        return None
    match = re.match(r"Confidence:\s*([0-9]+(?:\.[0-9]+)?)", raw[start:])
    return min(max(float(match.group(1)) / 100.0, 0.0), 1.0) if match else None


def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def decode_ids(tokenizer, ids, skip_special_tokens=True):
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens,
                            clean_up_tokenization_spaces=False)


def answer_span(gen_ids, processor, fmt):
    """以累计解码的字符偏移映射 token，过滤所有特殊 token。"""
    tokenizer = _tokenizer(processor)
    ids = [int(t) for t in gen_ids]
    raw = decode_ids(tokenizer, ids)
    meta = text_span(raw, fmt)
    special = set(tokenizer.all_special_ids)
    boundaries = [0]
    unstable_prefix = [False]
    for i in range(1, len(ids) + 1):
        prefix = decode_ids(tokenizer, ids[:i])
        # 字节 token 的未完成 UTF-8 字符可能临时解码为替换字符。
        common = 0
        for left, right in zip(prefix, raw):
            if left != right:
                break
            common += 1
        boundaries.append(max(boundaries[-1], common))
        unstable_prefix.append(common < len(prefix))
    start, end = meta["span_char_start"], meta["span_char_end"]
    selected = [i for i, token in enumerate(ids)
                if start < end and token not in special and boundaries[i] < end
                and boundaries[i + 1] > start]
    # 完整字符可能需要多个字节 token；回溯包含所有未完成前缀的 token。
    for i in list(selected):
        j = i - 1
        while j >= 0 and ids[j] not in special and unstable_prefix[j + 1]:
            selected.append(j)
            j -= 1
    selected = sorted(set(selected))
    prefix_extra = raw[boundaries[selected[0]]:start] if selected else ""
    suffix_extra = raw[end:boundaries[selected[-1] + 1]] if selected else ""
    partial = bool(prefix_extra.strip(' \t\r\n"') or suffix_extra.strip(' \t\r\n"'))
    interior = [i for i in selected if not (
        raw[boundaries[i]:start].strip(' \t\r\n"') if boundaries[i] < start else ""
    ) and not (
        raw[end:boundaries[i + 1]].strip(' \t\r\n"') if boundaries[i + 1] > end else ""
    )]
    meta.update({
        "raw": raw,
        "gen_ids": ids,
        "span_token_indices": selected,
        "span_token_ids": [ids[i] for i in selected],
        "span_len": len(selected),
        "span_interior_token_indices": interior,
        "span_boundary_prefix_extra": prefix_extra,
        "span_boundary_suffix_extra": suffix_extra,
        "span_boundary_whitespace_shared": bool((prefix_extra or suffix_extra) and not partial),
        "span_boundary_partial": partial,
        "span_probability_scope": "boundary_token_includes_nonanswer_text" if partial else (
            "answer_tokens_with_shared_whitespace" if prefix_extra or suffix_extra else "answer_tokens"),
        "span_token_char_boundaries": boundaries,
        "span_utf8_unstable_prefix_count": sum(unstable_prefix),
        "filtered_special_token_ids": [token for token in ids if token in special],
    })
    return meta


def confidence_from_logps(meta, token_logprobs):
    if len(token_logprobs) != len(meta["gen_ids"]):
        raise ValueError("生成 token 数与 logprob 数不一致")
    span_logps = [float(token_logprobs[i]) for i in meta["span_token_indices"]]
    if any(not math.isfinite(p) or p > 1e-6 for p in span_logps):
        raise ValueError("答案跨度包含无效 logprob")
    scoreable = bool(span_logps) and not meta.get("span_boundary_partial", False)
    interior_logps = [float(token_logprobs[i]) for i in meta.get(
        "span_interior_token_indices", meta["span_token_indices"])]
    return {
        "conf_token_geo": math.exp(sum(span_logps) / len(span_logps)) if scoreable else None,
        "conf_token_first": math.exp(span_logps[0]) if scoreable else None,
        "span_token_logprobs": span_logps,
        "span_scoreable": scoreable,
        "span_unscoreable_reason": None if scoreable else (
            "partial_boundary_token" if span_logps else "empty_answer_span"),
        "sensitivity_conf_token_geo_inclusive": math.exp(sum(span_logps) / len(span_logps)) if span_logps else None,
        "sensitivity_conf_token_first_inclusive": math.exp(span_logps[0]) if span_logps else None,
        "sensitivity_conf_token_geo_interior": math.exp(sum(interior_logps) / len(interior_logps)) if interior_logps else None,
        "sensitivity_span_len_interior": len(interior_logps),
    }


def score_generation(scores, gen_ids, processor, fmt):
    import torch
    if len(scores) != len(gen_ids):
        raise ValueError("生成 token 数与 scores 数不一致")
    logps = [torch.log_softmax(score[0].float(), dim=-1)[int(token)].item()
             for score, token in zip(scores, gen_ids)]
    meta = answer_span(gen_ids, processor, fmt)
    meta.update(confidence_from_logps(meta, logps))
    meta["gen_token_logprobs"] = logps
    meta["conf_verbalized"] = extract_verbalized(meta["raw"], fmt)
    return meta


def legacy_span_indices(gen_ids, tokenizer, fmt):
    """仅供历史证据对齐：精确复现修复前的跨度规则。"""
    base = base_format(fmt)
    start = 0
    if base != "A":
        cum = ""
        for i, token in enumerate(gen_ids):
            cum += tokenizer.decode([token], skip_special_tokens=False)
            if "Answer:" in cum:
                start = i + 1
                break
    end = len(gen_ids)
    if base == "C":
        cum = ""
        for i in range(start, len(gen_ids)):
            cum += tokenizer.decode([gen_ids[i]], skip_special_tokens=False)
            if "Confidence:" in cum:
                end = i
                break
    excluded = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    return [i for i in range(start, end) if gen_ids[i] not in excluded]
