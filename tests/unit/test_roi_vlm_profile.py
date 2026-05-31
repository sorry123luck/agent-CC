from src.vlm.roi_profile import roi_vlm_profile_for_job, roi_vlm_profile_for_jobs


def test_roi_vlm_profile_defaults_to_fast_small_payload():
    profile = roi_vlm_profile_for_jobs(
        [{"mode": "security_dashboard", "purpose": "feature_grid"}]
    )

    assert profile == {
        "profile": "fast",
        "max_candidate_ids": 4,
        "image_max_edge": 320,
        "max_tokens": 256,
    }


def test_roi_vlm_profile_expands_dense_control_matrix_budget():
    generic = roi_vlm_profile_for_jobs(
        [{"mode": "security_dashboard", "purpose": "feature_grid"}]
    )
    dense = roi_vlm_profile_for_jobs(
        [{"mode": "control_matrix", "purpose": "middle_control_columns"}]
    )

    assert dense["profile"] == "fast"
    assert dense["max_candidate_ids"] > generic["max_candidate_ids"]
    assert dense["max_tokens"] > generic["max_tokens"]
    assert dense["image_max_edge"] >= generic["image_max_edge"]


def test_roi_vlm_profile_expands_chat_and_collaboration_budget():
    chat = roi_vlm_profile_for_jobs(
        [{"mode": "chat_workspace", "purpose": "message_stream"}]
    )
    collaboration = roi_vlm_profile_for_jobs(
        [{"mode": "collaboration_inbox", "purpose": "inbox_list"}]
    )

    assert chat["max_candidate_ids"] == 4
    assert collaboration["max_candidate_ids"] == 4
    assert chat["max_tokens"] >= 384
    assert collaboration["max_tokens"] >= 384


def test_roi_vlm_profile_for_job_does_not_inflate_mixed_batch_neighbors():
    dense = roi_vlm_profile_for_job(
        {"mode": "control_matrix", "purpose": "middle_control_columns"}
    )
    generic = roi_vlm_profile_for_job(
        {"mode": "security_dashboard", "purpose": "feature_grid"}
    )

    assert dense["max_candidate_ids"] == 8
    assert generic["max_candidate_ids"] == 4
    assert generic["image_max_edge"] == 320
    assert generic["max_tokens"] == 256
