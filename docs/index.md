# MemEvolve Documentation

Welcome to the MemEvolve documentation! **MemEvolve is an API pipeline framework that proxies API requests to OpenAI compatible endpoints providing memory, memory management, and evolves the memory implementation thru mutations to enhance the memory system overtime.**

This comprehensive guide covers everything from getting started with the API proxy to understanding the self-evolving memory architecture.

## 📚 Documentation Structure

### 🚀 User Guide
Get started quickly and learn how to use MemEvolve in your applications.

- **[Getting Started](user-guide/getting-started.md)** - Quick setup and first steps
- **[Configuration](user-guide/configuration.md)** - Environment setup and options
- **[Deployment](user-guide/deployment.md)** - Docker and production deployment

### 🔧 API Reference
Technical details for developers integrating with MemEvolve.

- **[API Reference](api/api-reference.md)** - All endpoints and configuration options
- **[Troubleshooting](api/troubleshooting.md)** - Common issues and solutions

### 🛠️ Development
Technical documentation for contributors and advanced users.

- **[Architecture](development/architecture.md)** - System design and implementation
- **[Evolution System](development/evolution.md)** - Meta-evolution framework details
- **[Roadmap](development/roadmap.md)** - Development priorities and progress
- **[Development Scripts](development/scripts.md)** - Build and maintenance tools
- **[Performance Analyzer](tools/performance_analyzer.md)** - System monitoring and analysis
- **[Agent Guidelines](../AGENTS.md)** - Guidelines for coding agents

### 📖 Tutorials
Learn advanced patterns and best practices.

- **[Advanced Patterns](tutorials/advanced-patterns.md)** - Complex memory architectures

## 🎯 Quick Start

1. **For Users**: Start with [Getting Started](user-guide/getting-started.md)
2. **For Developers**: Check the [API Reference](api/api-reference.md)
3. **For Contributors**: Read [Agent Guidelines](../AGENTS.md)

## 📋 Key Topics

### API Pipeline Framework
- **Request Proxying**: Transparent interception of OpenAI-compatible API calls
- **Memory Injection**: Automatic context enhancement for all requests
- **Response Processing**: Experience extraction and memory storage
- **Zero Migration**: Drop-in replacement requiring no code changes

### Self-Evolving Memory System
- **Four Components**: Encode, Store, Retrieve, Manage working together
- **Meta-Evolution**: Memory architectures that evolve through mutations
- **Continuous Optimization**: System improves performance over time automatically
- **Production Safety**: Circuit breakers, monitoring, and rollback capabilities

### Memory Architecture Components
- **Encode**: Transforms API interactions into structured memories (lessons, skills, tools, abstractions)
- **Store**: Persists memories using vector databases for fast similarity search
- **Retrieve**: Finds relevant memories based on API request context
- **Manage**: Maintains memory health through pruning, consolidation, and deduplication

## 🔗 Related Resources

- **[GitHub Repository](https://github.com/thephimart/MemEvolve-API)**
- **[Issue Tracker](https://github.com/thephimart/MemEvolve-API/issues)**
- **[Research Paper](https://arxiv.org/abs/2512.18746)** - MemEvolve: Meta-Evolution of Agent Memory Systems

---

*Last updated: January 22, 2026*