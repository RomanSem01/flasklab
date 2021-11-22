var hoverCls = "file-hover";
var $fileDrop = $('[data-droppable=""]');
var $fileInput = $('[data-droppable-input=""]');
var $fileImage = $('[data-droppable-image=""]');

var fileHover = function(ev) {
  ev.stopPropagation();
  ev.preventDefault();
  if (ev.type === "dragover") {
    $fileDrop.addClass(hoverCls);
  } else {
    $fileDrop.removeClass(hoverCls);
  }
};

var fileSelect = function(ev) {
  fileHover(ev);
  var files = ev.target.files || ev.dataTransfer.files;
  var reader = new FileReader();

  reader.onload = function(ev) {
    $fileImage.css("background-image", "url(" + ev.target.result + ")");
  };
  reader.readAsDataURL(files[0]);
};

$fileDrop[0].ondragover = fileHover;
$fileDrop[0].ondragleave = fileHover;
$fileDrop[0].ondrop = fileSelect;
$fileInput[0].onchange = fileSelect;