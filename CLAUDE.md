# Deep Neural Network Learning

**Git Root**: I:\ai-automation-projects\deep-neural-network-learning

**Working Directory**: I:\ai-automation-projects\deep-neural-network-learning\

## Project Structure

- `.claude/rules/` - AI coding rules (PyTorch, testing, style)
- `.memo/` - Project documentation (specs, context)
- `.octie/` - Task management
- `phase1_basics/` - Neural network fundamentals
- `phase2_architectures/` - CNN, RNN, Transformer
- `phase3_training/` - Training techniques
- `phase4_advanced/` - Advanced training
- `phase5_deployment/` - Optimization & deployment

## Development Context

This is an educational deep learning implementation project with 30 hands-on tasks.

### Key Principles
1. Implement from scratch with NumPy first
2. Verify with PyTorch autograd
3. Pass gradient checks (< 1e-6 error)
4. Document theory + code + experiments

### Tech Stack
- Python 3.10+
- PyTorch 2.x
- NumPy 2.x
- pytest

## Commands

```bash
# Work in project directory
cd I:/ai-automation-projects/deep-neural-network-learning

# Run tests
pytest tests/ -v

# Format code
black .
```
