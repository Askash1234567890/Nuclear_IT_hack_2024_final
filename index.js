const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();

// создаем парсер для данных application/x-www-form-urlencoded
const urlencodedParser = express.urlencoded({extended: false});

// Слушаем порт 3000
app.listen(3000, () => console.log('Сервер запущен на http://localhost:3000'));

app.use(express.static(path.join(__dirname, '.')));

// Пример маршрутизации для обработки GET-запросов
app.get('/index.html', (req, res) => {
	res.sendFile(__dirname + '/index.html');
  });

  app.post("/index.html", urlencodedParser, function (request, response) {
    if(!request.body) return response.sendStatus(400);
	console.log(request.body.file);
    response.send(request.body.file);
});
