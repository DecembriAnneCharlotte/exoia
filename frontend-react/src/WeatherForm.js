import React, { useState } from "react";

const WeatherForm = () => {
  const [formData, setFormData] = useState({
    holiday: "",
    temp: "",
    rain_1h: "",
    snow_1h: "",
    clouds_all: "",
    weather_main: "",
    weather_description: "",
    year: "",
    month: "",
    day: "",
    hour: ""
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formDataToSend = {
      ...formData,
      temp: parseFloat(formData.temp), 
      rain_1h: formData.rain_1h ? parseFloat(formData.rain_1h) : null, 
      snow_1h: formData.snow_1h ? parseFloat(formData.snow_1h) : null,  
      clouds_all: parseInt(formData.clouds_all),  
  };

    try {
      const response = await fetch("http://34.155.24.177:8000/predict", {
      // const response = await fetch("http://localhost:8000/predict", {

        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formDataToSend), 
      });

      if (!response.ok) {
        throw new Error("Failed to submit data");
      }

      const data = await response.json();
      alert("Prediction du trafic avec ses conditions météos: " + data.prediction + " Véhicule sur la route");
    } catch (error) {
      console.error("Error submitting data:", error);
      alert("Failed to submit data.");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-blue-100">
    <form
      onSubmit={handleSubmit}
      className="w-full max-w-lg bg-white px-8 py-3 rounded-lg shadow-lg space-y-3"
    >
      <h2 className="text-xl font-bold text-gray-500 text-center">Prédiction du trafic en fonction de la météo</h2>

      {[
        { label: "Vacances", name: "holiday", type: "text" },
        { label: "Temperature", name: "temp", type: "number" },
        { label: "Pluie en 1h", name: "rain_1h", type: "number" },
        { label: "Neige en 1h", name: "snow_1h", type: "number" },
        { label: "Nuage (%)", name: "clouds_all", type: "number" },
        { label: "Météo globale", name: "weather_main", type: "text" },
        { label: "Météo description", name: "weather_description", type: "text" },
        { label: "Année", name: "year", type: "number" },
        { label: "Mois", name: "month", type: "number" },
        { label: "Jour", name: "day", type: "text" },
        { label: "Heure", name: "hour", type: "number" },
      ].map(({ label, name, type }) => (
        <div key={name} className="space-y-1">
          <label className="block text-xs font-bold text-gray-700">{label}</label>
          <input
            type={type}
            name={name}
            value={formData[name]}
            onChange={handleChange}
            className="w-full px-4 py-2 border rounded-md text-xs focus:outline-none focus:ring focus:ring-blue-200"
            required={name !== "rain_1h" && name !== "snow_1h"} 
          />
        </div>
      ))}

      <button
        type="submit"
        className="w-full bg-blue-400 text-white py-2 px-4 rounded-lg hover:bg-blue-300 focus:outline-none focus:ring focus:ring-blue-200"
      >
        Envoyer
      </button>
    </form>
  </div>
);
};

export default WeatherForm;