"""保守清洗：保留主题和正文，删除可明确识别的邮件头、引用及署名。"""

from collections import Counter
import re
import unicodedata

HEADER_FIELD = re.compile(r"^([A-Za-z][A-Za-z0-9_.\[\]-]*):[ \t]*(.*)$")
QUOTE_LINE = re.compile(r"^[ \t]*(?:(?:[|:][ \t]*)*>|[A-Za-z]{1,12}>[ \t]+)")
ATTRIBUTION_END = re.compile(r"\b(?:writes|wrote|says|said):[ \t]*$", re.IGNORECASE)
ATTRIBUTION_START = re.compile(r"^in[ \t]+(?:article\b|<)", re.IGNORECASE)
ON_DATE = re.compile(
    r"^on\b.*(?:\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|"
    r"july?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
    r"|\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b)", re.IGNORECASE,
)
EMAIL = re.compile(r"([A-Za-z0-9_.+-]+)@[A-Za-z0-9_.-]+")
SHORT_SIGNATURE = re.compile(r"^[ \t]*-([A-Za-z][A-Za-z0-9_.-]{2,30})[ \t]*$")
SENDER_LINE = re.compile(
    r"^(?:<?[A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+>?[ \t]*"
    r"(?:\([^)]*(?:\)|$)[ \t]*)?|"
    r"[A-Za-z .'-]{2,80}[<(][A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+[>)][ \t]*)"
    r"(?:(?:writes|wrote|says|said):[ \t]*)?$", re.IGNORECASE,
)
# 这是已实现流程的说明记录，不是通过改 True/False 就能启用的配置开关。
CLEANING_RULES = {
    "version": "current_author_news_v2",
    "unicode": "NFC",
    "remove_leading_mail_headers": True,
    "keep_subject": True,
    "remove_entire_quoted_lines": True,
    "remove_explicit_reply_attribution": True,
    "remove_author_matched_short_signoff": True,
    "preserve_unstructured_single_line": True,
    "lowercase": True,
    "collapse_whitespace": True,
    "truncate_at_signature_separator": False,
    "remove_stopwords": False,
    "stemming": False,
    "remove_all_punctuation": False,
    "remove_numbers_or_urls": False,
    "remove_html_with_generic_regex": False,
    "augmentation": False,
}


def remove_mail_headers(text):
    """只识别文首连续邮件头；保留 Subject，正文中的类似字段不删除。"""
    lines = text.split("\n")
    if not lines or not HEADER_FIELD.match(lines[0]):
        return text

    fields = {}
    current_key = None
    body_start = None
    for position, line in enumerate(lines):
        if not line.strip():
            body_start = position + 1
            break
        match = HEADER_FIELD.match(line)
        if match:
            current_key = match.group(1).lower()
            fields.setdefault(current_key, []).append(match.group(2))
        elif line.startswith((" ", "\t")) and current_key is not None:
            # 邮件头可折行，例如较长的 Subject。
            fields[current_key][-1] += " " + line.strip()
        else:
            # 不能确认为连续邮件头时，保留全文，避免误删正文。
            return text

    identity_fields = {"from", "message-id", "newsgroups", "organization"}
    if body_start is None or "subject" not in fields:
        return text
    if not identity_fields.intersection(fields):
        return text
    subject = " ".join(fields["subject"])
    body = "\n".join(lines[body_start:])
    return subject + "\n\n" + body


def author_signoffs(text):
    """从文首 From 确定署名，不使用验证/测试词频或人工作者名单。"""
    header = text.split("\n\n", 1)[0]
    match = re.search(r"(?im)^from:[ \t]*(.+)$", header)
    if not match:
        return set()
    from_value = match.group(1)
    email_names = {email.group(1).lower() for email in EMAIL.finditer(from_value)}
    names = re.search(r"\(([^()]+)\)", from_value)
    name_tokens = re.findall(r"[A-Za-z]+", names.group(1).lower()) if names else []
    candidates = email_names | {name for name in name_tokens if len(name) >= 3}
    if name_tokens:
        surname_initial = name_tokens[-1][0]
        # 常见邮箱形式为名字+姓氏首字母；andyh + Huang 可明确得到 andy。
        candidates |= {local[:-1] for local in email_names
                       if local.endswith(surname_initial) and len(local) >= 4}
    return candidates


def reply_attribution_length(lines, position):
    """识别至多五行的明确回复来源块；不吞掉来源之后的普通正文。"""
    start = lines[position].strip()
    possible_attribution = (
        ATTRIBUTION_START.match(start)
        or ON_DATE.match(start)
        or SENDER_LINE.fullmatch(start)
    )
    if not possible_attribution:
        return 0

    block = []
    for end in range(position, min(position + 5, len(lines))):
        candidate = lines[end].strip()
        if not candidate or QUOTE_LINE.match(lines[end]):
            break
        if end > position and not ATTRIBUTION_END.search(candidate):
            if not re.search(r"[<@()]", candidate):
                break  # 折行须仍带来源特征，不能将普通正文当作来源。
        block.append(candidate)
        joined = " ".join(block)
        if len(joined) > 700:
            break
        if ATTRIBUTION_END.search(joined):
            has_identity = bool(EMAIL.search(joined))
            dated_reply = bool(ON_DATE.match(start) and re.search(
                r"\b(?:writes|wrote):[ \t]*$", joined, re.IGNORECASE,
            ))
            return len(block) if has_identity or dated_reply else 0
    return 0


def remove_reply_history(text, signoffs, protect_subject=False):
    """逐行删除引用/来源/匹配署名；先处理行结构，再折叠空白。"""
    lines = text.split("\n")
    counts = Counter(quoted_lines_removed=0, attribution_lines_removed=0,
                     author_signoff_lines_removed=0)
    kept = []
    position = 0
    quote_context = False
    while position < len(lines):
        line = lines[position]
        if position == 0 and protect_subject:
            # 保留 Subject，即便标题本身形似引用或来源语句。
            kept.append(line)
            position += 1
            continue
        contextual_quote = quote_context and re.match(r"^[ \t]*[|:][ \t]+", line)
        if QUOTE_LINE.match(line) or contextual_quote:
            counts["quoted_lines_removed"] += 1
            quote_context = True
            position += 1
            continue

        attribution_length = reply_attribution_length(lines, position)
        if attribution_length:
            counts["attribution_lines_removed"] += attribution_length
            quote_context = True
            position += attribution_length
            continue

        signoff = SHORT_SIGNATURE.fullmatch(line)
        if signoff and signoff.group(1).lower() in signoffs:
            counts["author_signoff_lines_removed"] += 1
        else:
            kept.append(line)
            # 空行也结束单独 |/: 的引用上下文，避免吞掉后续作者解释。
            quote_context = False
        position += 1
    return "\n".join(kept), dict(counts)


def clean_with_trace(text):
    """固定、不拟合数据统计量的规则；训练、验证、最终测试共用。"""
    if not isinstance(text, str):
        raise TypeError("文本必须是字符串；不能把缺失值直接转换为 'nan'。")
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    body = remove_mail_headers(normalized)
    current_text, removed_lines = remove_reply_history(
        body, author_signoffs(normalized),
        # 单行无上下文时不猜测引用，避免重复清洗删掉 >80 等技术标题。
        protect_subject=body != normalized or "\n" not in body,
    )
    lowered = current_text.lower()
    cleaned = " ".join(lowered.split())
    flags = {
        "mail_headers_removed": body != normalized,
        "quoted_content_removed": removed_lines["quoted_lines_removed"] > 0,
        "reply_attribution_removed": removed_lines["attribution_lines_removed"] > 0,
        "author_signoff_removed": removed_lines["author_signoff_lines_removed"] > 0,
        "lowercase_changed": lowered != current_text,
        "whitespace_changed": cleaned != lowered,
    }
    flags.update(removed_lines)
    return cleaned, flags


def clean_text(text):
    """公开接口：最终预测阶段也应调用同一个函数。"""
    return clean_with_trace(text)[0]


