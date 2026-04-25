import { useState } from "react";

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const checkSpam = async () => {
    if (!message.trim()) {
      setError("Please enter a message");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
   const res = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ message }),
});

      if (!res.ok) {
        throw new Error("Server error. Try again.");
      }

      const data = await res.json();

      if (!data || data.confidence === undefined) {
        throw new Error("Invalid response from server");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-black via-purple-900 to-black text-white">
      
      <div className="w-full max-w-lg p-8 rounded-2xl bg-white/10 backdrop-blur-lg shadow-2xl border border-white/20">

        <h1 className="text-4xl font-bold text-center mb-2">
          Spam Detector
        </h1>
        <p className="text-center text-gray-300 mb-6">
          AI-powered SMS classifier
        </p>

        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message..."
          className="w-full p-4 rounded-lg bg-white/20 placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
        />

        <button
          onClick={checkSpam}
          disabled={loading}
          className={`mt-4 w-full py-3 rounded-lg font-semibold transition-all duration-300
            ${loading ? "bg-gray-500 cursor-not-allowed" : "bg-purple-600 hover:bg-purple-700"}
          `}
        >
          {loading ? "Analyzing..." : "Check Message"}
        </button>

        {/* 🔴 ERROR */}
        {error && (
          <div className="mt-4 text-red-400 text-center">
            {error}
          </div>
        )}

        {/* 🔄 LOADING SPINNER */}
        {loading && (
          <div className="flex justify-center mt-4">
            <div className="w-6 h-6 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
          </div>
        )}

        {/* ✅ RESULT */}
        {result && (
          <div className="mt-6 text-center transition-all duration-500">
            <h2
              className={`text-2xl font-bold ${
                result.prediction === "SPAM"
                  ? "text-red-400"
                  : "text-green-400"
              }`}
            >
              {result.prediction}
            </h2>

            <p className="mt-2 text-gray-300">
              Confidence: {(result.confidence * 100).toFixed(2)}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;