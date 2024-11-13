import re
from torch import FloatTensor
from dataclasses import dataclass
from typing import List, Dict, Any, Set, Tuple, Optional
from efficient_overshadowed_ed.data_types.baseclass import BaseDataType


@dataclass
class Entity(BaseDataType):
    identifier: str
    confident: Optional[float|FloatTensor] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"


@dataclass
class Span(BaseDataType):
    surface_form: str
    start: Optional[int] = None
    end: Optional[int] = None
    confident: Optional[float|FloatTensor] = None
    pred_entity: Optional[Entity] = None
    cand_entities: Optional[List[Entity]] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"
    
# matches = re.finditer(r'\$\d+\.\d{2}', text)

# for match in matches:
#     print(f"Match: {match.group()} at position {match.span()}")

@dataclass
class Passage(BaseDataType):
    text: str
    # confident: Optional[float|FloatTensor] = None
    entities: List[Span]
    # relevant_passages: Optional[List['Passage']] = None

    # Automatically get start and end index for each entity without having to manually input them
    def __post_init__(self):
        new_entities: List[Span] = []
        unique_entities: Set[Tuple] = set()
        for entity in self.entities:
            if entity.start is None or entity.end is None:
                matches = re.finditer(entity.surface_form, self.text)
                for match in matches:
                    new_entity = Span(
                        start=match.start(),
                        end=match.end(),
                        surface_form=entity.surface_form
                    )
                    if (new_entity.start, new_entity.end, new_entity.surface_form) not in unique_entities:
                        new_entities.append(new_entity)
                        unique_entities.add((new_entity.start, new_entity.end, new_entity.surface_form))
            else:
                if (entity.start, entity.end, entity.surface_form) not in unique_entities:
                    new_entities.append(entity)
                    unique_entities.add((entity.start, entity.end, entity.surface_form))
        self.entities = new_entities

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"


# @dataclass
# class Document(BaseDataType):
#     passages: List[Passage]
#     confident: Optional[float|FloatTensor] = None

#     def __repr__(self) -> str:
#         attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
#         return f"{self.__class__.__name__}({attributes})"

#     def __str__(self) -> str:
#         attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
#         return f"{self.__class__.__name__}({attributes})"