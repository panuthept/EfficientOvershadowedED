from efficient_overshadowed_ed.data_types import Passage, Span
from efficient_overshadowed_ed.model_wrappers import HuggingfaceReFinED


if __name__ == "__main__":
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
    for span in model(passages)[0].entities:
        print(span)