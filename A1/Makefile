.PHONY: setup doctor verify-sources test test-modal test-chess-modal test-docker test-chess-docker \
	llm-up llm-down check-part1 check-swebench \
	run-code-agent run-swebench-agent run-chess-agent \
	run-obs-experiment-no-legal-moves run-obs-experiment-legal-moves \
	run-obs-deepseek-no-legal run-obs-deepseek-legal \
	run-obs-gpt-oss-no-legal run-obs-gpt-oss-legal \
	run-chess-modal run-chess-sandbox instructor-eval-base instructor-eval-gold instructor-eval-patch

TASK ?= tasks/chess-terminal-move
CODE_SKILLS ?= tasks/code-skills
PATCH ?= artifacts/fix.patch
PART1_TRAJECTORY ?= artifacts/part1-trajectory.json
PRIVATE_EVAL ?= .private/chess-terminal-move
PUBLIC_EVAL ?= $(TASK)/public_tests
# One of the vendored instances under tasks/swebench/.
INSTANCE ?= django__django-15368
SWEBENCH_PATCH ?= artifacts/$(INSTANCE).patch
SWEBENCH_TRAJECTORY ?= artifacts/$(INSTANCE)-trajectory.json
TRAJECTORY ?= artifacts/part3-trajectory.json
RESULT ?= artifacts/game-result.json
MODEL ?= deepseek/deepseek-v4-flash-0731
DEEPSEEK_MODEL ?= deepseek/deepseek-v4-flash-0731
GPT_OSS_MODEL ?= openai/gpt-oss-120b
MODEL_TAG ?= $(subst /,-,$(MODEL))
COMPACT_THRESHOLD ?= 6000
STEPS ?= 200
CHESS_TIMEOUT ?= 1800
OBS_NO_LEGAL_TRAJECTORY ?= artifacts/part3-no-legal-moves-$(MODEL_TAG).json
OBS_LEGAL_TRAJECTORY ?= artifacts/part3-legal-moves-$(MODEL_TAG).json
OBS_NO_LEGAL_RESULT ?= artifacts/part3-no-legal-moves-$(MODEL_TAG)-result.json
OBS_LEGAL_RESULT ?= artifacts/part3-legal-moves-$(MODEL_TAG)-result.json

setup:
	uv sync
	git submodule update --init
	$(MAKE) verify-sources

doctor:
	uv run assignment-doctor

verify-sources:
	uv run python -c "from assignment.task import Task; from assignment.utils.image import verify_source; verify_source(Task.load('tasks/chess-terminal-move'))"

# Fast and free. These tests fail in the starter until the corresponding TODOs
# are implemented.
test:
	uv run pytest

# Slow but free: launches real local Docker sandboxes (the default backend).
test-docker:
	uv run pytest -m docker tests/test_docker_env.py

test-chess-docker:
	uv run pytest -m docker tests/test_docker_chess_sandbox.py

# Local OpenAI-compatible model server (vLLM in Docker) in place of the course
# endpoint. Prints the .env lines to use.
LLM_GPUS ?= 3
LLM_MODEL ?= $(GPT_OSS_MODEL)
llm-up:
	python3 scripts/local_llm.py up --gpus $(LLM_GPUS) --model $(LLM_MODEL)

llm-down:
	python3 scripts/local_llm.py down

# Slow and billable: launches real Modal sandboxes (SANDBOX_BACKEND=modal).
test-modal:
	SANDBOX_BACKEND=modal uv run pytest -m modal

test-chess-modal:
	SANDBOX_BACKEND=modal uv run pytest -m modal tests/test_chess_sandbox.py

# Apply the student's generated patch to a fresh testbed and run the public
# Part 1 regression test plus the existing chess-app suite.
check-part1:
	@test -f "$(PATCH)" || (echo "Patch not found: $(PATCH). Run make run-code-agent first."; exit 1)
	uv run python scripts/evaluate.py --task $(TASK) --evaluation $(PUBLIC_EVAL) \
		--patch $(PATCH) -v

# Apply a patch to the published SWE-bench image for INSTANCE and grade it
# against that instance's FAIL_TO_PASS and PASS_TO_PASS tests.
check-swebench:
	@test -f "$(SWEBENCH_PATCH)" || (echo "Patch not found: $(SWEBENCH_PATCH). Run make run-swebench-agent INSTANCE=$(INSTANCE) first."; exit 1)
	uv run python scripts/evaluate_swebench.py $(INSTANCE) --patch $(SWEBENCH_PATCH) -v

run-code-agent:
	uv run assignment-code-agent --task $(TASK) --model $(MODEL) --step-limit $(STEPS) \
		--skills-path $(CODE_SKILLS) --trajectory $(PART1_TRAJECTORY) \
		--patch-output $(PATCH)

# Run the code agent on a vendored SWE-bench instance, in its published image.
run-swebench-agent:
	uv run assignment-swebench-agent $(INSTANCE) --model $(MODEL) \
		--patch-output $(SWEBENCH_PATCH) \
		--trajectory $(SWEBENCH_TRAJECTORY) \
		--skills-path $(CODE_SKILLS) \
		--step-limit $(STEPS) $(if $(filter-out 0,$(COMPACT_THRESHOLD)),--compact-threshold-tokens $(COMPACT_THRESHOLD),)

run-chess-agent:
	uv run assignment-play-chess --model $(MODEL) --task $(TASK) --patch $(PATCH) \
		--step-limit $(STEPS) --sandbox-timeout $(CHESS_TIMEOUT) \
		--trajectory $(TRAJECTORY) --result $(RESULT)

run-obs-experiment-no-legal-moves:
	uv run assignment-play-chess --model $(MODEL) --task $(TASK) --patch $(PATCH) \
		--omit-legal-moves --step-limit $(STEPS) --sandbox-timeout $(CHESS_TIMEOUT) \
		--trajectory $(OBS_NO_LEGAL_TRAJECTORY) --result $(OBS_NO_LEGAL_RESULT)

run-obs-experiment-legal-moves:
	uv run assignment-play-chess --model $(MODEL) --task $(TASK) --patch $(PATCH) \
		--step-limit $(STEPS) --sandbox-timeout $(CHESS_TIMEOUT) \
		--trajectory $(OBS_LEGAL_TRAJECTORY) --result $(OBS_LEGAL_RESULT)

run-obs-deepseek-no-legal:
	$(MAKE) run-obs-experiment-no-legal-moves MODEL="$(DEEPSEEK_MODEL)" MODEL_TAG=deepseek

run-obs-deepseek-legal:
	$(MAKE) run-obs-experiment-legal-moves MODEL="$(DEEPSEEK_MODEL)" MODEL_TAG=deepseek

run-obs-gpt-oss-no-legal:
	$(MAKE) run-obs-experiment-no-legal-moves MODEL="$(GPT_OSS_MODEL)" MODEL_TAG=gpt-oss

run-obs-gpt-oss-legal:
	$(MAKE) run-obs-experiment-legal-moves MODEL="$(GPT_OSS_MODEL)" MODEL_TAG=gpt-oss

# Serves the chess app from a sandbox on the configured backend (Docker by default).
run-chess-sandbox: run-chess-modal
run-chess-modal:
	uv run assignment-chess-modal --task $(TASK) --sandbox-timeout $(CHESS_TIMEOUT) $(if $(wildcard $(PATCH)),--patch $(PATCH),)