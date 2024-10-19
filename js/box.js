// запоминаем область перетаскивания файла
const dropFileZone = document.querySelector(".upload-zone_dragover");
// запоминаем кнопку добавления файла
const uploadInput = document.querySelector(".form-upload__input");
// запоминаем кнопку отправки
const submitButton = document.querySelector('.form-upload__submit');
// записываем в переменную адрес, куда отправятся файлы после загрузки
const uploadUrl = "/uploadFile_Loader";

// добавляем обработчики событий "dragover" и "drop" для документа
["dragover", "drop"].forEach(function (event) {
    // блокируем стандартное поведение браузера для события и возвращаем false
    document.addEventListener(event, function (evt) {
      evt.preventDefault();
      return false;
    });
   });

// добавляем обработчик события для входа в зону перетаскивания файла"
dropFileZone.addEventListener("dragenter", function () {
    // добавляем класс стиля, красим форму
    dropFileZone.classList.add("_active");
   });

    // добавляем обработчик события для выхода из зоны перетаскивания файла"
    dropFileZone.addEventListener("dragleave", function () {
      // возвращаем цвет неактивной формы"
      dropFileZone.classList.remove("_active");
   });

// добавляем обработчик события "drop" для зоны перетаскивания"
dropFileZone.addEventListener("drop", function () {
    // удаляем класс активности при сбросе файла
    dropFileZone.classList.remove("_active");
    // получаем первый файл в списке
    const file = event.dataTransfer?.files[0];
    // проверяем, что файл есть
    if (file) {
      // готовим файл к отправке
      uploadInput.files = event.dataTransfer.files;
    }
   });

// добавляем обработчик события для файлов, добавленных кнопкой"
uploadInput.addEventListener("change", (event) => {
    // получаем первый файл в списке
    const file = uploadInput.files?.[0];
    // проверяем, что файл есть
    if (file) {
      // готовим файл к отправке
      uploadInput.files = event.dataTransfer.files;
    }
   });

// добавляем обработчик события "click" для кнопки отправки
submitButton.addEventListener("click", function (event) {
     // блокируем стандартное поведение кнопки (отправку формы)
    event.preventDefault();
    // вызываем функцию для отправки файла
    processingUploadFile();
   });

// функция для обработки загрузки файла
function processingUploadFile(file) {
  // проверяем, что файл был отправлен
  if (file) {
    // создаём объект для отправки данных формы
    const dropZoneData = new FormData();
    // создаём объект для отправки запроса на сервер
    const xhr = new XMLHttpRequest();

    // добавляем файл из формы в объект FormData
    dropZoneData.append("file", file);

    // открываем соединение с сервером
    xhr.open("POST", uploadUrl, true);

    // отправляем данные на сервер
    xhr.send(dropZoneData);

    // устанавливаем обработчик события onload для выполнения действий после завершения загрузки
      xhr.onload = function () {
      // проверяем статус ответа сервера
      if (xhr.status == 200) {
       // сообщаем об успехе в консоли"
       console.log("Всё загружено");
      } else {
       // соообщаем об ошибке в консоли"
       console.log("Ошибка загрузки");
      }
      // скрываем элемент
      HTMLElement.style.display = "none";
   };
 }
}
