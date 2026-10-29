"""
Telegram Bot for Yojana Sahayak.

Features:
- /start, /help, /clear, /schemes commands
- Inline scheme browser with per-field buttons (eligibility, benefits, how to apply)
- Text message handling — full RAG + LLM pipeline
- Voice message handling — MLX Whisper ASR → RAG + LLM → text reply
- Per-user conversation history (last 5 turns)
- Rate limiting (10 messages per minute per user)
- Webhook mode when WEBHOOK_URL is set; polling otherwise

Usage:
    python -m yojana_sahayak.bot.telegram_bot
    # or
    yojana-bot
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Rate limiting ─────────────────────────────────────────────────────────────

_user_timestamps: dict[int, list[float]] = defaultdict(list)
_RATE_LIMIT = 10  # max messages per 60 seconds per user


def _within_rate_limit(user_id: int) -> bool:
    now = time.monotonic()
    window = [t for t in _user_timestamps[user_id] if now - t < 60]
    _user_timestamps[user_id] = window
    if len(window) >= _RATE_LIMIT:
        return False
    _user_timestamps[user_id].append(now)
    return True


# ── Shared retriever singleton ────────────────────────────────────────────────

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from yojana_sahayak.rag.retriever import SchemeRetriever
        _retriever = SchemeRetriever()
        _retriever.build_index()
    return _retriever


# ── Scheme data for inline keyboard ──────────────────────────────────────────

POPULAR_SCHEMES = [
    "PM Kisan",
    "Ayushman Bharat",
    "PM Ujjwala Yojana",
    "Pradhan Mantri Mudra Yojana",
    "PM Awas Yojana",
    "Mahatma Gandhi National Rural Employment Guarantee",
    "Sukanya Samriddhi Yojana",
    "Atal Pension Yojana",
    "Pradhan Mantri Kaushal Vikas Yojana",
    "PM Jan Dhan Yojana",
]

POPULAR_SCHEME_LABELS = [
    "PM Kisan",
    "Ayushman Bharat",
    "PM Ujjwala",
    "PM Mudra Loan",
    "PM Awas",
    "MGNREGA",
    "Sukanya Samriddhi",
    "Atal Pension",
    "PMKVY",
    "Jan Dhan",
]

FIELD_LABELS = {
    "eligibility":         "✅ Eligibility / Patra kaun?",
    "benefits":            "💰 Benefits / Kya milega?",
    "application_process": "📝 How to Apply / Kaise karein?",
    "description":         "ℹ️ About / Kya hai?",
}

# ── Static text ───────────────────────────────────────────────────────────────

WELCOME_TEXT = (
    "🇮🇳 *Yojana Sahayak* — आपका सरकारी योजना सहायक\n\n"
    "किसी भी भारतीय सरकारी योजना के बारे में पूछें।\n"
    "Ask me about any Indian government scheme.\n\n"
    "*कैसे पूछें / How to ask:*\n"
    "• PM Kisan ke liye kaun eligible hai?\n"
    "• What are the benefits of Ayushman Bharat?\n"
    "• Ujjwala Yojana mein free gas kaise milega?\n"
    "• Mudra loan ke liye kya documents chahiye?\n\n"
    "*Commands:*\n"
    "/schemes — Popular schemes with quick\\-tap buttons\n"
    "/clear — नई बातचीत शुरू करें\n"
    "/help — Show this message\n\n"
    "💬 Hindi · English · Hinglish — सब चलता है\\!\n"
    "🎤 Voice messages also supported"
)

RATE_LIMIT_TEXT = (
    "⚠️ Thoda ruko\\! Ek minute mein itne messages mat bhejo\\.\n"
    "Please slow down — too many messages per minute\\."
)

ERROR_TEXT = (
    "⚠️ Kuch gadbad ho gayi\\. Please dobara try karein\\.\n"
    "Something went wrong\\. Please try again\\."
)

VOICE_FAIL_TEXT = (
    "🎤 Aawaz samajh nahi aayi\\. Please text mein likhein\\.\n"
    "Could not understand the audio\\. Please type your question\\."
)

NO_INFO_TEXT = (
    "Mujhe is bare mein specific jankari nahi mili\\. "
    "Aap [MyScheme Portal](https://www\\.myscheme\\.gov\\.in) par dekh sakte hain\\."
)


# ── Core pipeline (runs in a thread pool) ─────────────────────────────────────

def _run_pipeline(query: str, history: list) -> tuple[str, list]:
    """Synchronous RAG + LLM. Returns (answer, updated_history)."""
    from yojana_sahayak.asr.whisper import rewrite_query
    from yojana_sahayak.llm.generator import generate

    clean = rewrite_query(query)
    ctx = _get_retriever().retrieve_context(clean)
    answer = generate(clean, context=ctx, history=history)

    updated = list(history) + [{"user": clean, "assistant": answer}]
    return answer, updated[-5:]


def _transcribe(audio_path: str) -> str:
    """Synchronous: transcribe audio file using MLX Whisper (offline)."""
    wav_path = _to_wav(audio_path)
    try:
        from yojana_sahayak.asr.whisper import transcribe
        result = transcribe(wav_path)
    finally:
        if wav_path != audio_path and Path(wav_path).exists():
            os.unlink(wav_path)
    return result.get("text", "").strip()


def _to_wav(src: str) -> str:
    """Convert any audio to 16 kHz mono WAV via ffmpeg (already in Dockerfile)."""
    dst = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-i", src, "-ar", "16000", "-ac", "1", dst, "-y"],
        check=True, capture_output=True,
    )
    return dst


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_TEXT, parse_mode="MarkdownV2")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_TEXT, parse_mode="MarkdownV2")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    await update.message.reply_text(
        "✅ Conversation cleared\\. Naya sawaal poochh sakte hain\\!",
        parse_mode="MarkdownV2",
    )


async def cmd_schemes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = _schemes_keyboard()
    await update.message.reply_text(
        "🗂 *Popular Government Schemes*\nTap a scheme to explore:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


def _schemes_keyboard() -> list[list[InlineKeyboardButton]]:
    rows = []
    pair = []
    for label, full in zip(POPULAR_SCHEME_LABELS, POPULAR_SCHEMES):
        pair.append(InlineKeyboardButton(label, callback_data=f"s:{full}"))
        if len(pair) == 2:
            rows.append(pair)
            pair = []
    if pair:
        rows.append(pair)
    return rows


def _fields_keyboard(scheme: str) -> list[list[InlineKeyboardButton]]:
    rows = [
        [InlineKeyboardButton(label, callback_data=f"f:{scheme}:{field}")]
        for field, label in FIELD_LABELS.items()
    ]
    rows.append([InlineKeyboardButton("🔙 All schemes", callback_data="back")])
    return rows


# ── Inline keyboard callbacks ─────────────────────────────────────────────────

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "back":
        await query.edit_message_text(
            "🗂 *Popular Government Schemes*\nTap a scheme to explore:",
            reply_markup=InlineKeyboardMarkup(_schemes_keyboard()),
            parse_mode="Markdown",
        )

    elif data.startswith("s:"):
        scheme = data[2:]
        await query.edit_message_text(
            f"📋 *{scheme}*\n\nWhat would you like to know?",
            reply_markup=InlineKeyboardMarkup(_fields_keyboard(scheme)),
            parse_mode="Markdown",
        )

    elif data.startswith("f:"):
        _, scheme, field = data.split(":", 2)
        label = FIELD_LABELS.get(field, field)
        await query.edit_message_text(
            f"⏳ Looking up *{label}* for _{scheme}_…",
            parse_mode="Markdown",
        )
        loop = asyncio.get_running_loop()
        q = f"{scheme} {field}"
        history = list(context.user_data.get("history", []))
        try:
            answer, new_history = await loop.run_in_executor(
                None, _run_pipeline, q, history
            )
            context.user_data["history"] = new_history
            await query.edit_message_text(
                f"📋 *{scheme}*\n{label}\n\n{answer}",
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error("Callback pipeline error: %s", e, exc_info=True)
            await query.edit_message_text(
                f"📋 *{scheme}* — {label}\n\n⚠️ Error fetching info. Please try again.",
                parse_mode="Markdown",
            )


# ── Message handlers ──────────────────────────────────────────────────────────

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if not _within_rate_limit(user_id):
        await update.message.reply_text(RATE_LIMIT_TEXT, parse_mode="MarkdownV2")
        return

    text = update.message.text.strip()
    if not text:
        return

    await context.bot.send_chat_action(update.effective_chat.id, "typing")

    history = list(context.user_data.get("history", []))
    loop = asyncio.get_running_loop()

    try:
        answer, new_history = await loop.run_in_executor(
            None, _run_pipeline, text, history
        )
        context.user_data["history"] = new_history
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error("Text handler error: %s", e, exc_info=True)
        await update.message.reply_text(ERROR_TEXT, parse_mode="MarkdownV2")


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if not _within_rate_limit(user_id):
        await update.message.reply_text(RATE_LIMIT_TEXT, parse_mode="MarkdownV2")
        return

    await context.bot.send_chat_action(update.effective_chat.id, "typing")

    tmp_ogg = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp_path = tmp_ogg.name
    tmp_ogg.close()

    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        await voice_file.download_to_drive(tmp_path)

        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(None, _transcribe, tmp_path)

        if not transcript:
            await update.message.reply_text(VOICE_FAIL_TEXT, parse_mode="MarkdownV2")
            return

        # Echo transcript so the user knows what was heard
        await update.message.reply_text(
            f"🎤 *Suna:* {transcript}",
            parse_mode="Markdown",
        )
        await context.bot.send_chat_action(update.effective_chat.id, "typing")

        history = list(context.user_data.get("history", []))
        answer, new_history = await loop.run_in_executor(
            None, _run_pipeline, transcript, history
        )
        context.user_data["history"] = new_history
        await update.message.reply_text(answer)

    except Exception as e:
        logger.error("Voice handler error: %s", e, exc_info=True)
        await update.message.reply_text(
            "⚠️ Voice process nahi hua\\. Text mein likhein\\.",
            parse_mode="MarkdownV2",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── App builder ───────────────────────────────────────────────────────────────

def build_app() -> Application:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("schemes", cmd_schemes))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))

    return app


async def _register_commands(app: Application) -> None:
    await app.bot.set_my_commands([
        BotCommand("start",   "Welcome / शुरू करें"),
        BotCommand("schemes", "Browse popular schemes"),
        BotCommand("clear",   "Clear conversation history"),
        BotCommand("help",    "Show help"),
    ])


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logger.info("Pre-loading RAG index…")
    _get_retriever()

    app = build_app()

    # Register bot commands (best-effort)
    try:
        asyncio.get_event_loop().run_until_complete(_register_commands(app))
    except Exception:
        pass

    webhook_url = os.environ.get("WEBHOOK_URL", "").strip().rstrip("ß").strip()

    if webhook_url:
        port = int(os.environ.get("PORT", 8000))
        logger.info("Webhook mode: %s  port %d", webhook_url, port)
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            webhook_url=f"{webhook_url}/webhook",
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        logger.info("Polling mode — no WEBHOOK_URL set")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
