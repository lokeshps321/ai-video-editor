from app.prompt_parser import parse_prompt


def test_parse_trim_command() -> None:
    result = parse_prompt("trim clip 1 from 00:05 to 00:12")
    assert not result.errors
    assert len(result.operations) == 1
    op = result.operations[0]
    assert op.op_type == "trim_clip"
    assert op.params["clip"] == 1
    assert op.params["start_sec"] == 5
    assert op.params["end_sec"] == 12


def test_parse_export_command() -> None:
    result = parse_prompt("export 1080p 30fps high mp4")
    assert not result.errors
    op = result.operations[0]
    assert op.op_type == "set_export_settings"
    assert op.params["resolution"] == "1080p"
    assert op.params["fps"] == 30


def test_parse_invalid_prompt() -> None:
    result = parse_prompt("make this viral please")
    assert result.errors
    assert not result.operations
    assert result.suggestions


def test_parse_transition_and_crop_commands() -> None:
    transition = parse_prompt("transition clip 1 dissolve 0.6s")
    assert not transition.errors
    assert transition.operations[0].op_type == "set_transition"
    assert transition.operations[0].params["type"] == "dissolve"

    crop = parse_prompt("crop clip 2 0 0 720 1280")
    assert not crop.errors
    assert crop.operations[0].op_type == "crop_resize"
    assert crop.operations[0].params["width"] == 720


def test_parse_audio_fade_and_mute() -> None:
    fade = parse_prompt("audio fade out clip 1 1.5s")
    assert not fade.errors
    assert fade.operations[0].op_type == "set_volume"
    assert fade.operations[0].params["track_kind"] == "audio"
    assert fade.operations[0].params["fade_out_sec"] == 1.5

    mute = parse_prompt("mute clip 1")
    assert not mute.errors
    assert mute.operations[0].params["mute"] is True


def test_parse_track_and_move_commands() -> None:
    track_volume = parse_prompt("track audio volume 0.7")
    assert not track_volume.errors
    assert track_volume.operations[0].op_type == "set_volume"
    assert track_volume.operations[0].params["track_kind"] == "audio"
    assert track_volume.operations[0].params["volume"] == 0.7

    move = parse_prompt("move clip 2 to 00:05")
    assert not move.errors
    assert move.operations[0].op_type == "move_clip"
    assert move.operations[0].params["timeline_start_sec"] == 5


def test_parse_natural_language_variants() -> None:
    trim = parse_prompt("please trim first clip from 5 sec to 12 sec")
    assert not trim.errors
    assert trim.operations[0].op_type == "trim_clip"
    assert trim.operations[0].params["clip"] == 1
    assert trim.operations[0].params["start_sec"] == 5
    assert trim.operations[0].params["end_sec"] == 12

    merge = parse_prompt("join clips 1 2 3")
    assert not merge.errors
    assert merge.operations[0].op_type == "merge_clips"
    assert merge.operations[0].params["clips"] == [1, 2, 3]

    title = parse_prompt('add title "Hook text" at 00:01 for 2 sec')
    assert not title.errors
    assert title.operations[0].op_type == "add_text_overlay"
    assert title.operations[0].params["duration_sec"] == 2

    move = parse_prompt("shift clip #2 at 7 sec")
    assert not move.errors
    assert move.operations[0].op_type == "move_clip"
    assert move.operations[0].params["clip"] == 2
    assert move.operations[0].params["timeline_start_sec"] == 7

    export = parse_prompt("render 1080p 30 fps high quality mp4")
    assert not export.errors
    assert export.operations[0].op_type == "set_export_settings"
    assert export.operations[0].params["fps"] == 30
