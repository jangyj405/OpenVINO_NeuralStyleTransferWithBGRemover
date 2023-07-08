var imageFile;
var global_response;
function encodeImageFileAsURL(element) {
    var file = element.files[0];
    var reader = new FileReader();
    reader.onloadend = function () {
        console.log('RESULT', reader.result);
        imageFile = reader.result
        let img = document.getElementById('previewImage')
        img.src = imageFile
    }
    reader.readAsDataURL(file);
}

window.onload = ()=>{
    document.getElementById("filechooser").addEventListener('change', function (event) {
        imageFile = event.target.files[0];
        console.log(imageFile)
    })
}


function uploadFile_with_designhtml() {
    element_style = document.getElementById("style")
    element_bg = document.getElementById("bg")

    let style_value = $('input:radio[name=styleRadio]:checked').val();
    let bg_value = $('input:radio[name=bgRadio]:checked').val();


    console.log(style_value);
    console.log(bg_value);

    let formData = new FormData();
    if(imageFile == undefined){
        alert('select image file!')
        return
    }

    let watingpage = document.getElementById('waitingPage')
    watingpage.style.visibility = 'visible'
    let text = document.getElementById('timelog')
    let time = 0
    let interval = setInterval(()=>{
        time++
        text.innerText = time + " Seconds..."
    }, 1000)

    formData.append("image", imageFile);
    formData.append("style", style_value)
    formData.append("bg", bg_value)
    $.ajax({
        url: "/upload",
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            console.log(response)
            if(response == 'fail')
            {
                let watingpage = document.getElementById('waitingPage')
                watingpage.style.visibility = 'hidden'
                clearInterval(interval)
                alert('error! check image or try another!')
                return
            }
            let watingpage = document.getElementById('waitingPage')
            watingpage.style.visibility = 'hidden'
            clearInterval(interval)

            global_response = response
            let page = document.getElementById('resultPage')
            page.style.visibility = 'visible'
            let img = document.getElementById('resultImage')
            img.onload = ()=>{
                console.log(img.width)
                console.log(img.height)
                if(img.width > img.height)
                {
                    let ratio = 1.0*img.height/img.width
                    img.style.left = "20%"
                    img.style.width = "60%"
                    img.style.height = img.style.width * ratio
                }
                else
                {
                    let ratio = 1.0*img.width/img.height
                    img.style.height = "60%"
                    img.style.width = img.style.height * ratio
                    img.style.left = (100 - (60*ratio)/2).toFixed(1)
                }
            }
            img.src = "data:image/png;base64," + response

        },
        error: function (jqXHR, textStatus, errorMessage) {
            console.log(errorMessage); // Optional
            //error here
            let watingpage = document.getElementById('waitingPage')
            watingpage.style.visibility = 'hidden'
            clearInterval(interval)
            alert('error! check image or try another!')
        }
    });
}


function uploadFile() {
    element_style = document.getElementById("style")
    element_bg = document.getElementById("bg")
    var formData = new FormData();
    formData.append("image", imageFile);
    formData.append("style", element_style.value)
    formData.append("bg", element_bg.value)
    $.ajax({
        url: "/upload",
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            console.log(response)
            let img = document.createElement('img')
            img.src="data:image/png;base64,"+ response
            document.body.appendChild(img)
        },
        error: function (jqXHR, textStatus, errorMessage) {
            console.log(errorMessage); // Optional
        }
    });
}


function download_file(){
    var a = document.createElement("a"); //Create <a>
    a.href = "data:image/png;base64," + global_response; //Image Base64 Goes here
    a.download = "Image.png"; //File name Here
    a.click(); //Downloaded file
}

function for_test(){
    setInterval(() => {
        uploadFile_with_designhtml()    
    }, 300);
}