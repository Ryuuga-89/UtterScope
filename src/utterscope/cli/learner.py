"""CLI learner speaker selection."""

from __future__ import annotations

from utterscope.cli.console import console
from utterscope.cli.select import RadioOption, select_radio
from utterscope.diarization import SpeakerPreview


class InteractiveLearnerSelector:
    """Ask the user which of the detected speakers is the learner."""

    def select_learner(self, previews: list[SpeakerPreview]) -> str:
        if not previews:
            msg = "no speakers were detected in the transcript"
            raise ValueError(msg)
        if len(previews) == 1:
            speaker_id = previews[0].speaker_id
            console.print(
                f"\nOnly one speaker detected; using {speaker_id} as learner."
            )
            return speaker_id

        options = [
            RadioOption(
                label=preview.speaker_id,
                detail=f"{preview.speaking_time_seconds:.1f}s speaking",
                samples=tuple(preview.sample_texts),
            )
            for preview in previews
        ]
        console.print()
        index = select_radio(
            title="Which speaker is the learner (student)?",
            options=options,
        )
        return previews[index].speaker_id
