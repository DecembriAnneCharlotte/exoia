import React, { useEffect, useState } from "react";
import WeatherForm from "./WeatherForm";

function App() {
	const [message, setMessage] = useState("");

	useEffect(() => {
		fetch("http://localhost:8000/")

			.then((response) => response.json())
			.then((data) => setMessage(data.message));
	}, []);

	return (
		<div className="bg-blue-100">
			<WeatherForm />
			{/* <p>{message}</p> */}
		</div>
	);
}

export default App;
