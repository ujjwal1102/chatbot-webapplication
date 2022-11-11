

// const ws= JSON.parse(document.getElementById('room-name').textContent);

var ws = new WebSocket(
    'ws://'
    + window.location.host
    + '/ws/wsc/'
);
ws.onopen = function () {
    console.log('Websocket Connection Open...')
    var msg = document.getElementById('send-btn').value
    var msg = 'Hello'
    console.log(msg)
    user_message(msg)

    ws.send(msg)


}

ws.onmessage = function (event) {
    const data = JSON.parse(event.data)
    reply_message(data)

    console.log('Server says :- ', data.message)
    console.log("Count :- ", data.count)




}
ws.onerror = function (event) {
    console.log('Websocket Error Occured...', event)
}

ws.onclose = function (event) {
    console.log('Websocket Connection closed...', event)
}


