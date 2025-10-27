"""
title: Langfuse Filter Pipeline for v3 (Fixed)
author: open-webui (patched)
date: 2025-10-27
version: 0.0.2
license: MIT
description: Reliable Langfuse v3 integration with correct user/session tracking.
requirements: langfuse>=3.0.0
"""

from typing import List, Optional, Dict, Any
import os
import uuid
import json
from utils.pipelines.main import get_last_assistant_message
from pydantic import BaseModel
from langfuse import Langfuse


def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    """Retrieve the last assistant message from the message list."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        secret_key: str
        public_key: str
        host: str
        insert_tags: bool = True
        use_model_name_instead_of_id_for_generation: bool = False
        debug: bool = False

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter (Fixed)"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                "use_model_name_instead_of_id_for_generation": os.getenv("USE_MODEL_NAME", "false").lower() == "true",
                "debug": os.getenv("DEBUG_MODE", "false").lower() == "true",
            }
        )
        self.langfuse = None
        self.chat_traces: Dict[str, Any] = {}
        self.pending_chats: Dict[str, Dict] = {}  # Buffer until outlet
        self.model_names: Dict[str, Dict[str, str]] = {}
        self.suppressed_logs = set()

    def log(self, message: str, suppress_repeats: bool = False):
        if self.valves.debug:
            if suppress_repeats and message in self.suppressed_logs:
                return
            self.suppressed_logs.add(message)
            print(f"[Langfuse] {message}")

    async def on_startup(self):
        self.log("on_startup triggered")
        self.set_langfuse()

    async def on_shutdown(self):
        self.log("on_shutdown triggered")
        if self.langfuse:
            try:
                for chat_id, trace in self.chat_traces.items():
                    try:
                        trace.end()
                        self.log(f"Ended trace for chat_id: {chat_id}")
                    except Exception as e:
                        self.log(f"Failed to end trace {chat_id}: {e}")
                self.chat_traces.clear()
                self.langfuse.flush()
                self.log("All traces flushed on shutdown")
            except Exception as e:
                self.log(f"Flush failed: {e}")

    async def on_valves_updated(self):
        self.log("Valves updated → reinitializing Langfuse client")
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.log(f"Initializing Langfuse @ {self.valves.host}")
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=self.valves.debug,
            )
            self.langfuse.auth_check()
            self.log("Langfuse authenticated successfully")
        except Exception as e:
            self.log(f"Langfuse init failed: {e}")
            self.langfuse = None

    def _build_tags(self, task_name: str) -> list:
        tags = ["open-webui"]
        if self.valves.insert_tags and task_name not in ["user_response", "llm_response"]:
            tags.append(task_name)
        return tags

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("INLET called")

        if not self.langfuse:
            self.log("Langfuse not initialized → skipping")
            return body

        metadata = body.get("metadata", {})
        chat_id = metadata.get("chat_id", str(uuid.uuid4()))
        session_id = metadata.get("session_id")

        # Normalize chat_id for temp sessions
        if chat_id == "local" and session_id:
            chat_id = f"temporary-session-{session_id}"

        metadata["chat_id"] = chat_id
        body["metadata"] = metadata

        # Store model info
        model_id = body.get("model")
        model_info = metadata.get("model", {})
        if chat_id not in self.model_names:
            self.model_names[chat_id] = {"id": model_id}
        else:
            self.model_names[chat_id]["id"] = model_id
        if isinstance(model_info, dict) and "name" in model_info:
            self.model_names[chat_id]["name"] = model_info["name"]

        # === DO NOT CREATE TRACE HERE ===
        # Just buffer the first input
        if chat_id not in self.pending_chats:
            self.pending_chats[chat_id] = {
                "input": body["messages"],
                "metadata": metadata.copy(),
                "user": user,
                "model_id": model_id,
            }
            self.log(f"Buffered first message for chat_id: {chat_id}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("OUTLET called")

        if not self.langfuse:
            return body

        metadata = body.get("metadata", {})
        chat_id = metadata.get("chat_id")
        session_id = metadata.get("session_id")

        if chat_id == "local" and session_id:
            chat_id = f"temporary-session-{session_id}"

        if not chat_id:
            self.log("No chat_id in outlet → skipping")
            return body

        task_name = metadata.get("task", "llm_response")
        tags_list = self._build_tags(task_name)

        user_email = user.get("email") if user else None
        final_session_id = session_id or chat_id

        # === RECOVER FROM ANONYMOUS → AUTHENTICATED ===
        if chat_id in self.chat_traces:
            trace = self.chat_traces[chat_id]
            current_user_id = getattr(trace, "user_id", None)
            if current_user_id is None and user_email:
                self.log(f"User logged in mid-chat ({chat_id}) → ending anonymous trace")
                try:
                    trace.end()
                except:
                    pass
                del self.chat_traces[chat_id]

        # === CREATE TRACE ONLY NOW ===
        if chat_id not in self.chat_traces:
            pending = self.pending_chats.get(chat_id, {})
            input_messages = pending.get("input") or body.get("messages", [])
            pending_user = pending.get("user")
            pending_user_email = pending_user.get("email") if pending_user else None

            # Prefer current user, fallback to pending
            final_user_email = user_email or pending_user_email or "anonymous"

            trace_metadata = {
                **metadata,
                "user_id": final_user_email,
                "session_id": final_session_id,
                "interface": "open-webui",
                "task": task_name,
            }

            try:
                trace = self.langfuse.start_span(
                    name=f"chat:{chat_id}",
                    input=input_messages,
                    metadata=trace_metadata,
                )
                trace.update_trace(
                    user_id=final_user_email,
                    session_id=final_session_id,
                    tags=tags_list,
                    metadata=trace_metadata,
                )
                self.chat_traces[chat_id] = trace
                self.log(f"Trace created for chat_id: {chat_id}, user: {final_user_email}")
            except Exception as e:
                self.log(f"Failed to create trace: {e}")
                return body
            finally:
                self.pending_chats.pop(chat_id, None)
        else:
            trace = self.chat_traces[chat_id]

        # === LOG USER INPUT EVENT (if not already logged) ===
        if chat_id not in getattr(self, "_input_logged", set()):
            try:
                event_span = trace.start_span(
                    name=f"user_input:{uuid.uuid4()}",
                    input=body.get("messages", []),
                    metadata={"type": "user_input", "interface": "open-webui"},
                )
                event_span.end()
                self._input_logged = getattr(self, "_input_logged", set())
                self._input_logged.add(chat_id)
            except Exception as e:
                self.log(f"Failed to log user input: {e}")

        # === EXTRACT USAGE ===
        assistant_msg_obj = get_last_assistant_message_obj(body["messages"])
        usage = None
        if assistant_msg_obj:
            info = assistant_msg_obj.get("usage", {})
            if isinstance(info, dict):
                input_tokens = info.get("prompt_eval_count") or info.get("prompt_tokens")
                output_tokens = info.get("eval_count") or info.get("completion_tokens")
                if input_tokens is not None and output_tokens is not None:
                    usage = {"input": input_tokens, "output": output_tokens, "unit": "TOKENS"}

        # === UPDATE TRACE OUTPUT ===
        assistant_message = get_last_assistant_message(body["messages"])
        complete_metadata = {
            **metadata,
            "user_id": user_email or "anonymous",
            "session_id": final_session_id,
            "interface": "open-webui",
            "task": task_name,
        }
        try:
            trace.update_trace(
                output=assistant_message,
                metadata=complete_metadata,
                tags=tags_list,
            )
        except Exception as e:
            self.log(f"Failed to update trace output: {e}")

        # === LOG LLM GENERATION ===
        model_id = self.model_names.get(chat_id, {}).get("id", body.get("model"))
        model_name = self.model_names.get(chat_id, {}).get("name", "unknown")
        model_value = model_name if self.valves.use_model_name_instead_of_id_for_generation else model_id

        metadata["model_id"] = model_id
        metadata["model_name"] = model_name

        try:
            gen = trace.start_generation(
                name=f"llm_response:{uuid.uuid4()}",
                model=model_value,
                input=body["messages"],
                output=assistant_message,
                metadata={
                    **complete_metadata,
                    "type": "llm_response",
                    "model_id": model_id,
                    "model_name": model_name,
                },
            )
            if usage:
                gen.update(usage=usage)
            gen.end()
            self.log(f"Generation logged: {model_value}")
        except Exception as e:
            self.log(f"Failed to log generation: {e}")

        return body
