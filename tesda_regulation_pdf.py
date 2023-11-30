from typing import List, Optional
from pydantic import Field
from langchain.docstore.document import Document
from langchain.load.serializable import Serializable


class TesdaRegulationPDF(Serializable):
    name: str = Field(...)
    documents: List[Document]
    toc_page: Optional[int]
    core_pages: List[int] = []
    competency_map_pages: List[int] = []
    trainee_entry_requirements_pages: List[int] = []
    section1_pages: List[int] = []

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return (f"TesdaRegulationPDF(Name:{self.name}, Length: {len(self.documents)} documents, TOC page: {self.toc_page}, "
                f"Core pages: {self.core_pages}, Competency Map page: {self.competency_map_pages}, "
                f"Trainee Entry Requirements pages: {self.trainee_entry_requirements_pages}), "
                f"Section 1 pages: {self.section1_pages}")
