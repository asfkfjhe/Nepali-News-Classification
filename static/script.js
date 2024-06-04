async function classifyText() {
    const text = document.getElementById('text-input').value;
    
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
    });

    const data = await response.json();

    document.getElementById('gru-category').textContent = data.gru_prediction.category;
    document.getElementById('gru-probability').textContent = data.gru_prediction.probability.toFixed(2);
    
    document.getElementById('lstm-category').textContent = data.lstm_prediction.category;
    document.getElementById('lstm-probability').textContent = data.lstm_prediction.probability.toFixed(2);
}
