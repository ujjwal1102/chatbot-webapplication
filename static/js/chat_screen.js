

function user_message() {
    var t = new Date()
    var time = t.getHours() + ":" + t.getMinutes() + ":" + t.getSeconds()
    var time_shown = t.getHours() + ":" + t.getMinutes()
    const user_msg = document.getElementById("inputmsg").value;
    let msg = '<div  class="d-flex flex-row justify-content-end">' +
        '<div>' +
        '<p id="sendmsg" class="small p-2 me-3 mb-1  rounded-4  " style="background-color: #f5f6f7; font-family: Quicksand">' + user_msg + '</p>' +
        '<p class="small me-3 mb-1 rounded-4 text-muted d-flex justify-content-end" style="font-family: Quicksand" >' + time_shown + '</p>' +
        '</div >'
    //+ '<i class="text-white fa fa-thin fa-user p-2" ></i>'

    document.querySelector("#inputmsg").value = '';
    document.querySelector('#msg_screen').insertAdjacentHTML("beforeend", msg);
    ws.send(JSON.stringify(user_msg))
    scrollBottom()

}

function reply_message(data) {
    let msg =
        '<div class="d-flex flex-row justify-content-start ">' +
        // '<i class="text-white fa fa-thin fa-user p-2" ></i>' +
        '<div>' +
        '<p class="small p-2 ms-3 mb-1 rounded-4 bg-primary text-white" style=" font-family: Quicksand">' + data.message + '</p>' +
        '<p class="small ms-3 mb-1 rounded-4 text-muted style="font-family: Quicksand">' + data.current_time + '</p>' +
        '</div>' +
        '</div>'

    document.querySelector('#msg_screen').insertAdjacentHTML("beforeend", msg);
    scrollBottom();

}


function scrollBottom() {

    let msg_screen = document.querySelector('#msg_screen')
    msg_screen.scrollTo(0, msg_screen.scrollHeight)
}
