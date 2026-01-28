# MemEvolve Documentation

## 🚨 v2.0 Development Status

**IMPORTANT**: This documentation describes v2.0 in active development preparing for master branch merge. While core functionality is implemented and tested, **there are significant issues that need to be addressed before production deployment**.

### **Critical Issues Affecting Core Functionality**
- **Memory Encoding Verbosity**: All encoded memories contain verbose prefixes instead of direct insights (affects 100% of new memory creation)
- **Negative Token Efficiency**: Consistent -1000+ token losses per request due to unrealistic baseline calculations
- **Static Business Scoring**: All responses show identical business_value_score: 0.3 and roi_score: 0.1
- **Configuration Sync Failures**: Evolution settings don't propagate to runtime components

### **Development vs Production**
- **Use For Development**: Excellent for testing new features and understanding system capabilities
- **Do Not Deploy**: Critical issues must be resolved before production use
- **Track Progress**: See [dev_tasks.md](../dev_tasks.md) and [known issues](api/troubleshooting.md#known-issues-in-v20) for fix status

---

API pipeline framework that proxies requests to OpenAI-compatible endpoints, providing persistent memory and continuous architectural evolution.

## 📚 Documentation Structure

### User Guide
- **[Getting Started](user-guide/getting-started.md)** - Quick setup and first steps
- **[Configuration](user-guide/configuration.md)** - Environment setup (78 variables)
- **[Deployment Guide](user-guide/deployment_guide.md)** - Docker and production deployment
- **[Auto-Evolution](user-guide/auto-evolution.md)** - Multi-trigger automatic evolution
- **[Quality Scoring](user-guide/quality-scoring.md)** - Adaptive response quality evaluation

### API Reference
- **[API Reference](api/api-reference.md)** - Endpoints and configuration options
- **[Troubleshooting](api/troubleshooting.md)** - Common issues and solutions
- **[Quality Scoring](api/quality-scoring.md)** - Technical quality evaluation details
- **[Business Analytics](api/business-analytics.md)** - ROI and impact validation

### Development
- **[Architecture](development/architecture.md)** - System design and implementation
- **[Evolution System](development/evolution.md)** - Meta-evolution framework
- **[Roadmap](development/roadmap.md)** - Development priorities and progress
- **[Scripts](development/scripts.md)** - Build and maintenance tools
- **[Agent Guidelines](../AGENTS.md)** - Coding standards and guidelines

### Tools
- **[Performance Analyzer](tools/performance_analyzer.md)** - System monitoring and analysis
- **[Business Impact Analyzer](tools/business-impact-analyzer.md)** - ROI and validation tools

### Tutorials
- **[Advanced Patterns](tutorials/advanced-patterns.md)** - Complex memory architectures

## 🎯 Quick Start

### **v2.0 Development Notice**
1. **For Users**: Start with [Getting Started](user-guide/getting-started.md) - but **review known issues first**
2. **For Developers**: Check out [API Reference](api/api-reference.md) and [Known Issues](api/troubleshooting.md#known-issues-in-v20)
3. **For Contributors**: Read [Agent Guidelines](../AGENTS.md) and [dev_tasks.md](../dev_tasks.md) for current priorities

### **Development Workflow**
1. **Review Issues**: Check [troubleshooting guide](api/troubleshooting.md#known-issues-in-v20) for critical issues
2. **Test Features**: Use branch for development and testing only
3. **Monitor Progress**: Track fixes in [dev_tasks.md](../dev_tasks.md) and GitHub issues
4. **Contribute**: Help resolve critical issues to prepare for master branch merge

## 📋 Key Topics

### API Pipeline Framework
- **Request Proxying**: Transparent interception of OpenAI-compatible API calls
- **Memory Injection**: Automatic context enhancement for all requests
- **Adaptive Quality Scoring**: Historical context-based evaluation of response quality
- **Response Processing**: Experience extraction and memory storage
- **Business Analytics**: Executive-level ROI tracking and impact validation
- **Zero Migration**: Drop-in replacement requiring no code changes

### Self-Evolving Memory System
- **Four Components**: Encode, Store, Retrieve, Manage working together
- **Intelligent Auto-Evolution**: Multi-trigger automatic evolution (requests, performance, plateau, time)
- **Business Impact Validation**: Statistical significance testing and ROI measurement
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

---

**⚠️ Version 2.0 Development Notice**: This documentation describes a development version preparing for master branch merge. Critical functionality issues exist and must be resolved before production deployment. See [Known Issues](api/troubleshooting.md#known-issues-in-v20) for current status.

*Last updated: January 28, 2026*