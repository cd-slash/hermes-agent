from pathlib import Path

import yaml


MERGE_CHANGED_IF = "steps.merge.outputs.changed == 'true'"


def _load_workflow():
    workflow_path = (
        Path(__file__).resolve().parents[1] / ".github/workflows/upstream-sync.yml"
    )
    assert workflow_path.exists(), f"missing workflow: {workflow_path}"
    with workflow_path.open("r", encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)
    return workflow, workflow_path


def _workflow_on(workflow):
    return workflow.get("on", workflow.get(True, {}))


def _job_steps(job):
    return {step["name"]: step for step in job["steps"]}


def test_upstream_sync_workflow_triggers_and_permissions():
    workflow, _ = _load_workflow()

    on_config = _workflow_on(workflow)
    assert on_config["schedule"] == [{"cron": "*/15 * * * *"}]
    assert "workflow_dispatch" in on_config
    assert workflow["permissions"]["contents"] == "write"
    assert workflow["concurrency"] == {
        "group": "upstream-sync",
        "cancel-in-progress": False,
    }


def test_upstream_sync_workflow_merges_verifies_and_pushes():
    workflow, workflow_path = _load_workflow()

    job = workflow["jobs"]["sync-upstream"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 15
    steps = _job_steps(job)

    checkout_step = steps["Checkout code"]
    assert (
        checkout_step["uses"]
        == "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5"
    )
    assert checkout_step["with"]["fetch-depth"] == 0
    assert checkout_step["with"]["ref"] == "main"

    setup_uv_step = steps["Install uv"]
    assert (
        setup_uv_step["uses"]
        == "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"
    )
    assert setup_uv_step["if"] == MERGE_CHANGED_IF

    python_step = steps["Set up Python 3.11"]
    assert python_step["if"] == MERGE_CHANGED_IF
    assert "uv python install 3.11" in python_step["run"]

    install_step = steps["Install dependencies"]
    assert install_step["if"] == MERGE_CHANGED_IF
    assert "uv venv .venv --python 3.11" in install_step["run"]
    assert "source .venv/bin/activate" in install_step["run"]
    assert "python -m ensurepip --upgrade --default-pip" in install_step["run"]
    assert 'uv pip install -e ".[all,dev]"' in install_step["run"]

    git_identity_step = steps["Configure git author"]
    assert "if" not in git_identity_step
    assert 'git config user.name "github-actions[bot]"' in git_identity_step["run"]
    assert (
        'git config user.email "41898282+github-actions[bot]@users.noreply.github.com"'
        in git_identity_step["run"]
    )

    ensure_upstream_step = steps["Ensure upstream remote"]
    assert "if" not in ensure_upstream_step
    assert (
        "git remote add upstream https://github.com/NousResearch/hermes-agent.git"
        in ensure_upstream_step["run"]
    )
    assert (
        "git remote set-url upstream https://github.com/NousResearch/hermes-agent.git"
        in ensure_upstream_step["run"]
    )

    merge_step = steps["Fetch and merge upstream/main"]
    assert merge_step["id"] == "merge"
    assert "if" not in merge_step
    assert "git fetch upstream main" in merge_step["run"]
    assert "git merge-base --is-ancestor upstream/main HEAD" in merge_step["run"]
    assert "git diff --quiet HEAD upstream/main" not in merge_step["run"]
    assert "git merge --no-edit upstream/main" in merge_step["run"]
    assert "changed=true" in merge_step["run"]

    verification_step = steps["Run focused verification"]
    assert verification_step["if"] == MERGE_CHANGED_IF
    assert (
        "scripts/run_tests.sh tests/plugins/memory/test_mem0_plugin.py tests/test_upstream_sync_workflow.py -q"
        in verification_step["run"]
    )

    push_step = steps["Push merged main"]
    assert push_step["if"] == MERGE_CHANGED_IF
    assert "git push origin HEAD:main" in push_step["run"]

    assert workflow_path.name == "upstream-sync.yml"
