from src.integration.enhancement_policy import full_local_enhancement_strategy


class Canvas:
    def __init__(self, *, mode: str, warnings: list[str]):
        self.artifacts = {
            "visual_pattern": {"mode": mode},
            "perception_quality": {"warnings": warnings},
        }


def test_uses_lightweight_uia_vision_for_sparse_collaboration_inbox():
    canvas = Canvas(mode="collaboration_inbox", warnings=["sparse_elements", "coarse_regions"])

    assert full_local_enhancement_strategy(canvas) == "lightweight_uia_vision"


def test_uses_full_local_enhancement_for_chat_workspace():
    canvas = Canvas(mode="chat_workspace", warnings=["unknown_role_heavy"])

    assert full_local_enhancement_strategy(canvas) == "full"
