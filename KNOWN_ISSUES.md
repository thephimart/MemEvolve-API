# Known Issues

This document tracks known issues and limitations in MemEvolve that are currently unresolved or require workarounds.

## 🚨 Critical Issues

### Evolution System Errors

**Issue**: Evolution system throws errors when enabled via `MEMEVOLVE_ENABLE_EVOLUTION=true`

**Symptoms**:
- EvolutionManager fails to apply genotype configurations
- `_apply_genotype_to_memory_system()` method doesn't actually reconfigure memory components
- Fitness evaluation tests the same unchanged system regardless of genotype
- System logs show "Failed to apply genotype" errors

**Workaround**: Keep evolution disabled (`MEMEVOLVE_ENABLE_EVOLUTION=false`) until Phase 5 evolution fixes are implemented.

**Status**: High priority fix planned for Phase 5 development.

## ⚠️ Provider-Specific Issues

### GLM-4.6V-Flash Compatibility Issues

**Issue**: GLM-4.6V-Flash model has output formatting issues when proxied through MemEvolve

**Problems**:
1. **No reasoning/content split**: Model output doesn't properly separate reasoning from final content, causing parsing errors in memory encoding
2. **Language reversion**: Model tends to revert to Chinese output when proxied, even when prompted in English

**Symptoms**:
- Memory encoding fails with parsing errors
- Inconsistent language in responses (mix of English prompts, Chinese responses)
- Context injection may not work properly due to malformed outputs

**Workaround**:
- Use alternative models for memory encoding (`MEMEVOLVE_LLM_BASE_URL` pointing to a different model)
- Disable memory integration for GLM-4.6V-Flash (`MEMEVOLVE_API_MEMORY_INTEGRATION=false`)
- Monitor responses for language consistency issues

**Status**: No fix planned - currently unusable with MemEvolve. Use alternative models.

## 🔧 Memory System Limitations

### Large Memory Performance Degradation

**Issue**: Performance degrades significantly with large memory databases (>10,000 units)

**Symptoms**:
- Increased latency on memory retrieval (>500ms)
- Higher CPU usage during search operations
- Potential memory leaks in long-running processes

**Workaround**:
- Enable auto-pruning: `MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true`
- Set reasonable limits: `MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=5000`
- Use vector storage backend for better performance with large datasets

**Status**: Known limitation. Performance optimization needed for enterprise-scale deployments.

### JSON Storage Concurrency Issues

**Issue**: JSON file storage can have concurrency issues with multiple simultaneous requests

**Symptoms**:
- Race conditions when multiple processes access memory.json simultaneously
- Potential data corruption in high-throughput scenarios
- File locking errors in multi-threaded environments

**Workaround**:
- Use single-threaded deployment for JSON storage
- Switch to vector storage backend for concurrent access: `MEMEVOLVE_STORAGE_BACKEND_TYPE=vector`
- Implement file locking mechanisms if JSON storage must be used

**Status**: Limitation of JSON storage backend. Consider database backends for production use.

## 🌐 API Compatibility Issues

### Provider-Specific Streaming Support

**Issue**: Some LLM providers may have limited streaming compatibility

**Symptoms**:
- Provider-specific streaming format variations
- Potential compatibility issues with certain models

**Workaround**: Test streaming with your specific provider; fall back to non-streaming if issues occur.

**Status**: Streaming support is implemented but may vary by provider.

## 📊 Monitoring and Observability Gaps

### Limited Error Reporting

**Issue**: Error messages and logging could be more informative for troubleshooting

**Symptoms**:
- Generic error messages that don't pinpoint root causes
- Insufficient debug information for complex issues
- Difficulty diagnosing configuration problems

**Workaround**:
- Enable debug logging: `MEMEVOLVE_LOG_LEVEL=DEBUG`
- Check individual component logs: `MEMEVOLVE_LOG_API_SERVER_ENABLE=true`

**Status**: Error reporting improvements planned for Phase 2.

## 🔒 Security Considerations

### API Key Exposure Risk

**Issue**: API keys may be exposed in logs when debug logging is enabled

**Symptoms**:
- Sensitive API keys appear in log files
- Potential security risk in shared environments

**Workaround**:
- Disable detailed logging in production: `MEMEVOLVE_LOG_LEVEL=WARNING`
- Use environment variables instead of config files for sensitive data

**Status**: Security hardening planned for Phase 2 (IMPORTANT priority).

## 📝 Contributing and Reporting

If you encounter an issue not listed here:

1. Check if it's covered in the [Troubleshooting Guide](docs/troubleshooting.md)
2. Search existing [GitHub Issues](https://github.com/thephimart/memevolve/issues)
3. Create a new issue with detailed information

---

*Last updated: January 21, 2026*</content>
</xai:function_call">/dev/null