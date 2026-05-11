let currentFilter = 'all';
let historyData = [];

function showTab(tabName, event) {
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });

    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');

    if (tabName === 'history') {
        loadHistory();
    }
}

async function analyzeInstruction() {
    const input = document.getElementById('instruction-input').value.trim();
    if (!input) {
        alert('Please enter some text to analyze.');
        return;
    }

    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const button = document.querySelector('#analyzer button');

    loading.style.display = 'block';
    result.style.display = 'none';
    button.disabled = true;

    try {
        const response = await fetch('/analyze_paragraph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: input })
        });

        const data = await response.json();
        loading.style.display = 'none';
        result.style.display = 'block';
        button.disabled = false;

        const resultsDiv = document.getElementById('analysis-results');
        if (data.error) {
            resultsDiv.innerHTML = `<p class="error-message">${data.error}</p>`;
            return;
        }

        let html = '<ul class="result-list">';
        data.forEach(item => {
            const className = item.prediction === 'instruction' ? 'result-item instruction' : 'result-item not-instruction';
            html += `<li class="${className}">
                <strong>${item.prediction.toUpperCase()}</strong> (${(item.confidence * 100).toFixed(1)}% confidence)<br>
                ${item.sentence}
            </li>`;
        });
        html += '</ul>';
        resultsDiv.innerHTML = html;
        document.getElementById('instruction-input').value = '';

        if (document.getElementById('history').classList.contains('active')) {
            loadHistory();
        }

    } catch (error) {
        loading.style.display = 'none';
        button.disabled = false;
        alert('Error analyzing text. Please try again.');
        console.error('Error:', error);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/history');
        historyData = await response.json();
        displayHistory(historyData);
    } catch (error) {
        console.error('Error loading history:', error);
        document.getElementById('history-list').innerHTML = '<div class="no-history"><p>Error loading history.</p></div>';
    }
}

function displayHistory(history) {
    const historyList = document.getElementById('history-list');

    if (!history || history.length === 0) {
        historyList.innerHTML = '<div class="no-history"><p>No prediction history yet. Try analyzing some instructions first!</p></div>';
        return;
    }

    const filteredHistory = history.filter(item => {
        const matchesFilter = currentFilter === 'all' || item.prediction === currentFilter;
        const matchesSearch = !document.getElementById('search-input').value ||
            item.text.toLowerCase().includes(document.getElementById('search-input').value.toLowerCase());
        return matchesFilter && matchesSearch;
    });

    if (filteredHistory.length === 0) {
        historyList.innerHTML = '<div class="no-history"><p>No items match your current filters.</p></div>';
        return;
    }

    const historyHtml = filteredHistory.map(item => `
        <div class="history-item">
            <div class="history-text">"${item.text}"</div>
            <div class="history-meta">
                <span class="history-prediction ${item.prediction.replace(' ', '-')}">${item.prediction.toUpperCase()}</span>
                <span>Confidence: ${(item.confidence * 100).toFixed(1)}%</span>
                <span>${item.datetime}</span>
            </div>
        </div>
    `).join('');

    historyList.innerHTML = historyHtml;
}

function setFilter(filter, event) {
    currentFilter = filter;
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    displayHistory(historyData);
}

function filterHistory() {
    displayHistory(historyData);
}

async function clearHistory() {
    if (confirm('Are you sure you want to clear all prediction history? This action cannot be undone.')) {
        try {
            const response = await fetch('/clear_history', {
                method: 'POST'
            });

            if (response.ok) {
                historyData = [];
                displayHistory(historyData);
                alert('History cleared successfully!');
            } else {
                alert('Error clearing history.');
            }
        } catch (error) {
            console.error('Error clearing history:', error);
            alert('Error clearing history.');
        }
    }
}

window.onload = async function () {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        document.getElementById('train-accuracy').textContent = `${(data.accuracy * 100).toFixed(1)}%`;
    } catch (error) {
        document.getElementById('train-accuracy').textContent = 'N/A';
    }
};

document.getElementById('instruction-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        analyzeInstruction();
    }
});
