import os

from typing import Optional, List, Dict, Any
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


class PIIDetector:
    MAX_CHARS_PER_DOC = 5120  # Azure's document character limit
    MAX_BATCH_SIZE = 5  # Max documents per request (free/standard tier)

    def __init__(self, language: str = "en"):
        self.language = language

        tac_endpoint = os.environ["PIIDETECTOR_TAC_ENDPOINT"]
        tac_key = os.environ["PIIDETECTOR_TAC_KEY"]
        self.client = TextAnalyticsClient(endpoint=tac_endpoint, credential=AzureKeyCredential(tac_key))

    def detect(
        self, text: str, categories_filter: Optional[List[str]] = None, domain_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        chunks = self._chunk(text)
        redacted_parts: List[str] = []
        all_entities: List[Dict[str, Any]] = []

        # Send in batches to respect the API limit
        for batch_start in range(0, len(chunks), self.MAX_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + self.MAX_BATCH_SIZE]
            response = self.client.recognize_pii_entities(
                batch, language=self.language, categories_filter=categories_filter, domain_filter=domain_filter
            )

            for i, doc in enumerate(response):
                chunk_offset = (batch_start + i) * self.MAX_CHARS_PER_DOC

                if doc.is_error:
                    raise RuntimeError(f"Azure error: {doc.error}")

                redacted_parts.append(doc.redacted_text)
                for entity in doc.entities:
                    all_entities.append(
                        {
                            "text": entity.text,
                            "category": entity.category,
                            "subcategory": entity.subcategory,
                            "confidence_score": entity.confidence_score,
                            "offset": entity.offset + chunk_offset,
                            "length": entity.length,
                        }
                    )

        return {"redacted_text": "".join(redacted_parts), "entities": all_entities}

    def detect_file(self, path: str, **kwargs) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return self.detect(f.read(), **kwargs)

    def _chunk(self, text: str) -> List[str]:
        if len(text) <= self.MAX_CHARS_PER_DOC:
            return [text]
        return [text[i : i + self.MAX_CHARS_PER_DOC] for i in range(0, len(text), self.MAX_CHARS_PER_DOC)]
