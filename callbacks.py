from typing import Dict, Any, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallBackHandler(BaseCallbackHandler):
    def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        print("""Runs when llm starts running""")
        print(f"***Prompt to LLM was: ***\n {prompts[0]}")
        print('*******')

    def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
    ) -> Any:
        print("""Runs when llm stops running""")
        print(f"***LLM Response: ***\n {response.generations[0][0].text}")
        print('*******')
