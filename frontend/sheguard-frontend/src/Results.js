function Results({ tfPrediction, visionAnalysis }) {
  return (
    <div>
      <h2>Analysis Results</h2>
      <p>Deepfake Prediction: {tfPrediction}</p>
      <p>Adult Content: {visionAnalysis.adult}</p>
      <p>Violence: {visionAnalysis.violence}</p>
      <p>Racy Content: {visionAnalysis.racy}</p>
    </div>
  );
}