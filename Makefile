# Makefile

C_SRC_EXTENSIONS := c cpp h hpp
C_SRC_DIRS := ./dynamo_/include ./dynamo_/src ./src/eva/bindings
C_IGNORED_FILES := ./dynamo_/include/UTL.hpp ./path/to/bad_file.cpp  # <- this is a placeholder
C_SRC_FILES := $(shell find $(C_SRC_DIRS) -type f \( $(foreach ext,$(C_SRC_EXTENSIONS), -iname '*.$(ext)' -o ) -false \))
C_FORMATTABLE_FILES := $(filter-out $(C_IGNORED_FILES),$(C_SRC_FILES))

.PHONY: style show-files check-style test clean clean-logs clean-cache gen-docs

style:
	@echo "Formatting Python files... 💅"
	@find . -name '*.py' -exec ruff format --config pyproject.toml {} + -o -name '*.pyi' -exec ruff format --config pyproject.toml {} +
	@echo "Formatting C/CPP files... 💅"
	@$(foreach file,$(C_FORMATTABLE_FILES), clang-format -i $(file) && echo "✨ Formatted: $(file)";)
	@echo "Formatting done! 💖"

show-files:
	@echo "📂 Files to be formatted:"
	@$(foreach file,$(C_FORMATTABLE_FILES), echo $(file);)

# Check formatting without making changes
check-style:
	@echo "Checking Python formatting..."
	@find . -name '*.py' -exec ruff format --config pyproject.toml --diff --quiet {} +

# Add other project tasks below...
test:
	@pytest tests/

clean:
	@echo "Cleaning temporary files and build artifacts..."
	# Remove Python cache directories and bytecode
	rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache .coverage .tox
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -delete -o -name '*.pyo' -delete

	# Remove Vim swap files and directories
	find . -name '*.swp' -delete -o -name '*.swo' -delete
	find . -type d -name ".vim" -exec rm -rf {} +

	# Remove build artifacts and package directories
	rm -rf dist/ build/* docs/build/* *.egg-info/ .eggs/

	# Remove IDE/editor-specific files
	# rm -rf .idea/ .vscode/ .cache/

	# Remove macOS-specific files
	find . -name '.DS_Store' -delete

	# Remove log files
	find . -name '*.log' -delete

	# Remove lib symlink files
	rm -rf src/eva/lib/*

	# Remove temporary files
	rm -f *~ .*~ *.bak .*.bak

	# Remove debug files
	# @find . -name 'test' -delete
	# @find . -name 'workflow' -delete

	@echo "Clean complete!"

clean-logs:
	# Remove all logs dirs
	find . -type d -name "logs" -exec rm -rf {} +
	find . -type d -name "log" -exec rm -rf {} +

	# Remove all log files
	find . -name '*.log' -delete

	@echo "Clean complete!"

clean-cache:
	# Remove Python cache directories
	find . -type d -name '__pycache__' -exec rm -rf {} +

	# Remove macOS-specific files
	find . -name '.DS_Store' -delete

	@echo "Clean complete!"

gen-docs:
	cd docs && sphinx-apidoc -o source/ ../src/
	cd docs && make html

# Makefile ends here
