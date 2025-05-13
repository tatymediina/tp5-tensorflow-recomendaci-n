// ! Array de las bandas y su respectivo género
const bands = [
  { name: "Metallica", genre: "metal" },
  { name: "Slayer", genre: "metal" },
  { name: "Taylor Swift", genre: "pop" },
  { name: "Ariana Grande", genre: "pop" },
  { name: "Radiohead", genre: "rock" },
  { name: "Nirvana", genre: "rock" },
  {name: "Stray Kids", genre:"kpop"},
  {name: "BTS", genre:"kpop"},
  {name: "The 1975", genre:"indie"},
  {name: "The Neighbourhood", genre:"indie"}
];

// ! Mostrar los grupos al usuario
const bandsContainer = document.getElementById("bandsContainer");
bands.forEach((band, i) => {
  const div = document.createElement("div");
  div.classList.add("band");
  div.innerHTML = `
    <label for="band${i}">${band.name}</label>
    <input type="number" id="band${i}" min="1" max="10" />
  `;
  bandsContainer.appendChild(div);
});

// ! Función para procesar los datos del usuario usando Tensorflow.js
function processData() {
  const inputs = []; // * entradas para el modelo
  const outputs = []; // * salidas para el modelo

  // ! Mapa que convierte cada género en un vector one-hot
  const genreMap = {
    metal: [1, 0, 0,0,0],
    pop: [0, 1, 0,0,0],
    rock: [0, 0, 1,0,0],
    kpop: [0, 0, 0, 1,0],
    indie: [0, 0, 0,0,1]
  };

  // ! Recorremos cada banda para obtener la calificación del usuario
  bands.forEach((band, i) => {
    const rating = parseInt(document.getElementById(`band${i}`).value); // * leeemos la calificación
    if (!isNaN(rating)) {  //solo si es un número válido
      inputs.push(genreMap[band.genre]); //*  Codificamos el género y lo guardamos como entrada
      outputs.push([rating / 10]); //*  Normalizamos la calificación a un valor entre 0-1
    }
  });

  const inputTensor = tf.tensor2d(inputs);
  const outputTensor = tf.tensor2d(outputs);

  // ! Modelo simple
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [5], units: 5, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

  model.fit(inputTensor, outputTensor, { epochs: 200 }).then(() => {
    
    // ! Generar predicciones para todos los géneros
    const predictions = bands.map((band, i) => {
      const genreVec = tf.tensor2d([genreMap[band.genre]]); // creamos el vector del género
      const pred = model.predict(genreVec).dataSync()[0];// obtenemos la predicción
      return { name: band.name, score: pred }; // guardamos el nombre y el puntaje
    });

    // ! Ordenar y mostrar recomendaciones
    const sorted = predictions.sort((a, b) => b.score - a.score);
    const ul = document.getElementById("recommendations");
    ul.innerHTML = "";
    sorted.forEach(b => {
      const li = document.createElement("li");
      li.textContent = `${b.name} (Puntaje estimado: ${(b.score * 10).toFixed(1)})`;
      ul.appendChild(li);
    });
  });
}
