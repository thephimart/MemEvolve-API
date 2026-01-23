// Theme management
function initTheme() {
    const savedTheme = localStorage.getItem('dashboard-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('dashboard-theme', newTheme);
}

// Data management
function refreshData() {
    fetch('/dashboard-data')
        .then(response => response.json())
        .then(data => {
            updateUI(data);
        })
        .catch(error => {
            console.error('Error refreshing data:', error);
            showError('Failed to refresh dashboard data');
        });
}

function updateUI(data) {
    document.getElementById('last-updated').textContent = 'Last updated: ' + data.timestamp;

    // Update system health
    const systemHealth = data.system_health || {};
    updateMetric('system-status', (systemHealth.status || 'healthy').toUpperCase());
    updateMetric('system-uptime', (systemHealth.uptime_percentage || 100) + '%');

    // Update API performance
    const apiPerf = data.api_performance || {};
    updateMetric('api-requests', apiPerf.total_requests || 0);
    updateMetric('api-success-rate', (apiPerf.success_rate || 0).toFixed(1) + '%');
    updateMetric('api-avg-time', (apiPerf.avg_response_time || 0).toFixed(2) + 's');

    // Update memory system
    const memSys = data.memory_system || {};
    updateMetric('memory-experiences', memSys.total_experiences || 0);
    updateMetric('memory-retrievals', memSys.retrieval_count || 0);
    updateMetric('memory-avg-time', (memSys.avg_retrieval_time || 0).toFixed(3) + 's');
    updateMetric('memory-utilization', (memSys.utilization || 0).toFixed(1) + '%');
    updateMetric('memory-size', (memSys.file_size_kb || 0).toFixed(1) + ' KB');

    // Update evolution system
    const evoSys = data.evolution_system || {};
    updateMetric('evolution-status', evoSys.status || 'Inactive');
    updateMetric('evolution-genotype', (evoSys.current_genotype || 'None').substring(0, 8));
    updateMetric('evolution-generations', evoSys.generations_completed || 0);
    updateMetric('evolution-fitness', (evoSys.fitness_score || 0).toFixed(4));
    updateMetric('evolution-quality', (evoSys.response_quality_score || 0).toFixed(3));

    // Update trends
    const trends = data.performance_trends || {};
    const trendStatus = trends.fitness_trend || 'stable';
    updateMetric('trend-status', trendStatus.charAt(0).toUpperCase() + trendStatus.slice(1));

    // Update trend status classes
    const trendElement = document.getElementById('trend-status');
    trendElement.className = 'metric-value trend-' + trendStatus;

    // Update resources
    const resources = data.resource_usage || {};
    updateMetric('resource-logs', (resources.log_storage_mb || 0).toFixed(1) + ' MB');
    updateMetric('resource-memory', (resources.memory_file_size_kb || 0).toFixed(1) + ' KB');
    updateMetric('resource-growth', (resources.log_growth_rate_kb_per_day || 0).toFixed(1) + ' KB/day');
}

function updateMetric(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function showError(message) {
    // Could implement a toast notification system here
    console.error(message);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initTheme();
    refreshData();

    // Auto-refresh every 30 seconds
    setInterval(refreshData, 30000);
});