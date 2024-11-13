# Efficient Overshadowed Entity Disambiguation by Mitigating Shortcut Learning

## Installation
```
git clone https://github.com/panuthept/EfficientOvershadowedED.git
cd EfficientOvershadowedED

conda create -n cft python=3.11.4
conda activate cft
pip install -e .
```

## Usage
```python
from efficient_overshadowed_ed.data_types import Passage, Span
from efficient_overshadowed_ed.model_wrappers import HuggingfaceReFinED


model = HuggingfaceReFinED.from_pretrained("panuthept/CFT-ReFinED")
passages = [
    Passage(
        text='I absolutely love the MCU movies, but Spider-Man said it best in Civil War when he saw Cap throwing his shield and said, "That thing doesn’t obey the laws of physics at all."',
        entities=[
            Span(surface_form="MCU"),
            Span(surface_form="Spider-Man"),
            Span(surface_form="Civil War"),
            Span(surface_form="Cap"),
            Span(surface_form="shield"),
        ]
    ),
]
print(model(passages))
>> [
    Passage(
        text='I absolutely love the MCU movies, but Spider-Man said it best in Civil War when he saw Cap throwing his shield and said, "That thing doesn’t obey the laws of physics at all."', 
        entities=[
            Span(surface_form='MCU', start=22, end=25, entity=Entity(identifier='Q642878', confident=0.9823)),
            Span(surface_form='Spider-Man', start=38, end=48, entity=Entity(identifier='Q79037', confident=0.9956)),
            Span(surface_form='Civil War', start=65, end=74, entity=Entity(identifier='Q726495', confident=0.6984)),
            Span(surface_form='Cap', start=87, end=90, entity=Entity(identifier='Q190679', confident=0.9035)),
            Span(surface_form='shield', start=104, end=110, entity=Entity(identifier='Q690141', confident=0.4816)),
        ]
    )
]
```