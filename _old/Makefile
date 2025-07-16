# HSOF Makefile
# Build automation for HSOF project

.PHONY: all build test docs clean validate benchmark install dev-setup

# Default target
all: validate build test

# Install dependencies
install:
	@echo "📦 Installing Julia dependencies..."
	julia --project=. -e 'using Pkg; Pkg.instantiate()'
	@echo "✓ Dependencies installed"

# Validate environment
validate:
	@echo "🔍 Validating environment..."
	@julia validate_environment.jl

# Build project
build:
	@echo "🔧 Building HSOF..."
	@julia build.jl --kernels

# Build documentation
docs:
	@echo "📚 Building documentation..."
	@julia build.jl --docs

# Run tests
test:
	@echo "🧪 Running tests..."
	@julia --project=. -e 'using Pkg; Pkg.test()'

# Run quick tests
test-quick:
	@echo "🧪 Running quick tests..."
	@julia --project=. test/cuda_validation.jl
	@julia --project=. test/test_config_loader.jl

# Run GPU tests
test-gpu:
	@echo "🧪 Running GPU tests..."
	@julia --project=. test/gpu/run_kernel_tests.jl

# Run benchmarks
benchmark:
	@echo "📊 Running benchmarks..."
	@julia --project=. benchmarks/run_benchmarks.jl

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@julia build.jl --clean
	@rm -rf docs/build/
	@rm -rf .julia/compiled/
	@echo "✓ Clean complete"

# Development setup
dev-setup: install
	@echo "🛠️  Setting up development environment..."
	@# Install pre-commit hooks
	@cp scripts/pre-commit .git/hooks/pre-commit 2>/dev/null || echo "No pre-commit hook found"
	@chmod +x .git/hooks/pre-commit 2>/dev/null || true
	@# Create local config files
	@for config in configs/*.toml.example; do \
		target=$${config%.example}; \
		if [ ! -f "$$target" ]; then \
			cp "$$config" "$$target"; \
			echo "Created $$target"; \
		fi \
	done
	@# Set up environment variables
	@if [ ! -f .env ]; then \
		echo "JULIA_NUM_THREADS=auto" > .env; \
		echo "JULIA_CUDA_MEMORY_POOL=cuda" >> .env; \
		echo "JULIA_CUDA_SOFT_MEMORY_LIMIT=0.9" >> .env; \
		echo "Created .env file"; \
	fi
	@echo "✓ Development setup complete"

# Format code
format:
	@echo "🎨 Formatting Julia code..."
	@julia --project=. -e 'using JuliaFormatter; format("src", verbose=true)'
	@julia --project=. -e 'using JuliaFormatter; format("test", verbose=true)'

# Lint code
lint:
	@echo "🔍 Linting code..."
	@julia --project=. scripts/lint.jl

# Start Julia REPL with project
repl:
	@julia --project=. --startup-file=startup.jl

# Run specific GPU device
gpu0:
	@CUDA_VISIBLE_DEVICES=0 julia --project=.

gpu1:
	@CUDA_VISIBLE_DEVICES=1 julia --project=.

# Performance profiling
profile:
	@echo "📊 Running performance profiling..."
	@julia --project=. scripts/profile.jl

# Generate performance report
perf-report:
	@echo "📈 Generating performance report..."
	@julia --project=. scripts/generate_perf_report.jl

# Docker targets
docker-build:
	@echo "🐳 Building Docker image..."
	@docker build -t hsof:latest .

docker-run:
	@echo "🐳 Running HSOF in Docker..."
	@docker run --gpus all -it --rm \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/configs:/workspace/configs \
		-v $(PWD)/results:/workspace/results \
		hsof:latest

# Help target
help:
	@echo "HSOF Makefile Targets:"
	@echo "  make all         - Validate, build, and test"
	@echo "  make install     - Install Julia dependencies"
	@echo "  make validate    - Validate environment"
	@echo "  make build       - Build CUDA kernels"
	@echo "  make docs        - Build documentation"
	@echo "  make test        - Run all tests"
	@echo "  make test-quick  - Run quick tests"
	@echo "  make test-gpu    - Run GPU tests"
	@echo "  make benchmark   - Run benchmarks"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make dev-setup   - Set up development environment"
	@echo "  make format      - Format code"
	@echo "  make lint        - Lint code"
	@echo "  make repl        - Start Julia REPL"
	@echo "  make gpu0/gpu1   - Run with specific GPU"
	@echo "  make profile     - Run performance profiling"
	@echo "  make help        - Show this help"

# Variables
JULIA ?= julia
JULIA_PROJECT = --project=.
JULIA_FLAGS = --color=yes

# CI targets
ci-test:
	@$(JULIA) $(JULIA_PROJECT) $(JULIA_FLAGS) -e 'using Pkg; Pkg.test(coverage=true)'

ci-coverage:
	@$(JULIA) $(JULIA_PROJECT) -e 'using Pkg; Pkg.add("Coverage"); using Coverage; Codecov.submit(process_folder())'