from pydantic import BaseModel, Field
from typing import List

class Reflection(BaseModel):
    missing : str = Field(description="Cretique of what is missing.")
    superflouous : str = Field(description="Critique of what is superflouous")

class Answer(BaseModel):

    """Answer the Question"""

    answer : str = Field(description="Around 250 word detailed answer of the question")
    reflection : Reflection = Field(description="your thoughts on the inital answer")
    search_queries : list = Field(description="1-3 Search queries for researching improvements to address the critique of your answer")

class ReviserAnswer(Answer):
    """Use the previous critique and revise your pervious answer"""
    references : List[str] = Field(description="Citations motivating your updated answer")