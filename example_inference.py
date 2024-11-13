from efficient_overshadowed_ed.data_types import Passage, Span
from efficient_overshadowed_ed.model_wrappers import HuggingfaceReFinED


if __name__ == "__main__":
    model = HuggingfaceReFinED.from_pretrained("panuthept/CFT-ReFinED")

    passages = [
        Passage(
            text='I absolutely love the MCU movies, but Spider-Man said it best in Civil War when he saw Cap throwing his shield and said, "That thing doesn’t obey the laws of physics at all."',
            entities=[
                Span(surface_form="shield"),
            ]
        ),
    ]
    passages = model(passages)
    print(passages)