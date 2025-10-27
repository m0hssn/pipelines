"""
title: Langfuse Filter Pipeline for v3
author: open-webui
date: 2025-07-31
version: 0.0.1
license: MIT
description: A filter pipeline that uses Langfuse v3.
requirements: langfuse>=3.0.0
"""

from typing import List, Optional
import os
import uuid
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
        self.name = "Langfuse Filter"
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
        self.chat_traces = {}
        self.suppressed_logs = set()
        self.model_names = {}

    def log(self, message: str, suppress_repeats: bool = False):
        if self.valves.debug:
            if suppress_repeats:
                if message in self.suppressed_logs:
                    return
                self.suppressed_logs.add(message)
            print(f"[DEBUG] {message}")

    async def on_startup(self):
        self.log(f"on_startup triggered for {__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        self.log(f"on_shutdown triggered for {__name__}")
        if self.langfuse:
            try:
                for chat_id, trace in self.chat_traces.items():
                    try:
                        trace.end()
                        self.log(f"Ended trace on shutdown for chat_id: {chat_id}")
                    except Exception as e:
                        self.log(f"Failed to end trace on shutdown {chat_id}: {e}")
                self.chat_traces.clear()
            except Exception as e:
                self.log(f"Error during shutdown: {e}")

    async def on_valves_updated(self):
        self.log("Valves updated, resetting Langfuse client.")
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.log(f"Initializing Langfuse with host: {self.valves.host}")
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=self.valves.debug,
            )
            try:
                self.langfuse.auth_check()
                self.log(f"Langfuse authenticated successfully at {self.valves.host}")
            except Exception as e:
                self.log(f"Auth check failed: {e}")
                self.langfuse = None
        except Exception as e:
            self.log(f"Langfuse initialization failed: {e}")
            self.langfuse = None

    def _build_tags(self, task_name: str) -> list:
        tags_list = []
        if self.valves.insert_tags:
            tags_list.append("open-webui")
            if task_name not in ["user_response", "llm_response"]:
                tags_list.append(task_name)
        return tags_list

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("Langfuse Filter INLET called")
        if not self.langfuse:
            self.log("[WARNING] Langfuse client not initialized - Skipped")
            return body

        metadata = body.get("metadata", {})
        chat_id = metadata.get("chat_id", str(uuid.uuid4()))
        if chat_id == "local":
            session_id = metadata.get("session_id")
            chat_id = f"temporary-session-{session_id}"
        metadata["chat_id"] = chat_id
        body["metadata"] = metadata

        model_info = metadata.get("model", {})
        model_id = body.get("model")
        if chat_id not in self.model_names:
            self.model_names[chat_id] = {"id": model_id}
        else:
            self.model_names[chat_id]["id"] = model_id
        if isinstance(model_info, dict) and "name" in model_info:
            self.model_names[chat_id]["name"] = model_info["name"]

        required_keys = ["model", "messages"]
        if not all(k in body for k in required_keys):
            self.log("Error: Missing required keys in body")
            return body

        user_email = user.get("email") if user else None
        task_name = metadata.get("task", "user_response")
        tags_list = self._build_tags(task_name)

        trace_metadata = {
            **metadata,
            "user_id": user_email,
            "session_id": chat_id,
            "interface": "open-webui",
        }

        if chat_id not in self.chat_traces:
            self.log(f"Creating new trace for chat_id: {chat_id}")
            try:
                trace = self.langfuse.trace(
                    name=f"chat:{chat_id}",
                    user_id=user_email,
                    session_id=chat_id,
                    metadata=trace_metadata,
                    tags=tags_list,
                    input=body,
                )
                self.chat_traces[chat_id] = trace
                self.log(f"Trace created for chat_id: {chat_id}")
            except Exception as e:
                self.log(f"Failed to create trace: {e}")
                return body
        else:
            trace = self.chat_traces[chat_id]
            trace.update(
                user_id=user_email,
                session_id=chat_id,
                metadata=trace_metadata,
                tags=tags_list,
            )

        # Log user input as span
        try:
            span = trace.span(
                name=f"user_input:{str(uuid.uuid4())}",
                input=body["messages"],
                metadata={**trace_metadata, "type": "user_input"}
            )
            span.end()
        except Exception as e:
            self.log(f"Failed to log user input span: {e}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("Langfuse Filter OUTLET called")
        if not self.langfuse:
            self.log("[WARNING] Langfuse client not initialized - Skipped")
            return body

        chat_id = body.get("chat_id")
        if chat_id == "local":
            session_id = body.get("session_id")
            chat_id = f"temporary-session-{session_id}"

        if chat_id not in self.chat_traces:
            self.log(f"[WARNING] No trace found for chat_id: {chat_id}, re-running inlet")
            return await self.inlet(body, user)

        trace = self.chat_traces[chat_id]
        metadata = body.get("metadata", {})
        task_name = metadata.get("task", "llm_response")
        tags_list = self._build_tags(task_name)

        assistant_message = get_last_assistant_message(body["messages"])
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])

        # Extract usage
        usage = None
        if assistant_message_obj:
            info = assistant_message_obj.get("usage", {})
            if isinstance(info, dict):
                input_tokens = info.get("prompt_eval_count") or info.get("prompt_tokens")
                output_tokens = info.get("eval_count") or info.get("completion_tokens")
                if input_tokens is not None and output_tokens is not None:
                    usage = {"input": input_tokens, "output": output_tokens, "unit": "TOKENS"}

        user_email = user.get("email") if user else None
        complete_metadata = {
            **metadata,
            "user_id": user_email,
            "session_id": chat_id,
            "interface": "open-webui",
            "task": task_name,
        }

        # Update trace with output and final metadata
        trace.update(
            output=assistant_message,
            metadata=complete_metadata,
            tags=tags_list,
        )

        # Determine model for generation
        model_id = self.model_names.get(chat_id, {}).get("id", body.get("model"))
        model_name = self.model_names.get(chat_id, {}).get("name", "unknown")
        model_value = model_name if self.valves.use_model_name_instead_of_id_for_generation else model_id

        metadata["model_id"] = model_id
        metadata["model_name"] = model_name

        # Create generation
        try:
            generation = trace.generation(
                name=f"llm_response:{str(uuid.uuid4())}",
                model=model_value,
                input=body["messages"],
                output=assistant_message,
                metadata={**complete_metadata, "type": "llm_response", "generation_id": str(uuid.uuid4())},
                usage=usage,
            )
            generation.end()
            self.log(f"Generation completed and ended for chat_id: {chat_id}")
        except Exception as e:
            self.log(f"Failed to create generation: {e}")

        # CRITICAL FIX: End the trace immediately after outlet
        try:
            trace.end()
            self.log(f"Trace ended for chat_id: {chat_id}")
            # Optional: remove from active traces to prevent reuse
            self.chat_traces.pop(chat_id, None)
        except Exception as e:
            self.log(f"Failed to end trace: {e}")

        return body
