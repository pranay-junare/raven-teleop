//////////////////////////////////////////////////
/////     USER INTERACTION SUPPORT ROUTINES
//////////////////////////////////////////////////
const ws = new WebSocket("ws://localhost:8080");

ws.onopen = () => console.log("Connected to ZMQ bridge");



kineval.initKeyEvents = function init_keyboard_events() {
    document.addEventListener('keydown', function(e) {
        kineval.handleKeydown(e.keyCode);
    }, true);
}

kineval.handleKeydown = function handle_keydown(keycode) {
    switch (keycode) {
    case 74: // j 
        kineval.changeActiveLinkDown();
        break;
    case 75: // k
        kineval.changeActiveLinkUp();
        break;
    case 76: // l
        kineval.changeActiveLinkNext();
        break;
    case 72: // h
        kineval.changeActiveLinkPrevious();
        break;
    case 84: // t
        kineval.toggleStartpointMode();
        break;
    case 37: // left arrow
        rosCmdVel.publish(rosTwistLft);
        console.log('trying to move left');
        break;
    case 38: // up arrow
        rosCmdVel.publish(rosTwistFwd);
        console.log('trying to move forward');
        break;
    case 39: // right arrow
        rosCmdVel.publish(rosTwistRht);
        console.log('trying to move right');
        break;
    case 40: // down arrow
        rosCmdVel.publish(rosTwistBwd);
        console.log('trying to move backward');
        break;
    case 13: // enter
        rosManip.publish(rosManipGrasp);
        console.log('trying to grasp');
        break;
    }
}

kineval.handleUserInput = function user_input() {
    if (!robot || !robot.origin || !robot.origin.xyz || robot.origin.xyz.length < 3) return; // safe check

    if (keyboard.pressed("z")) {
        camera.position.x += 0.1*(robot.origin.xyz[0] - camera.position.x);
        camera.position.y += 0.1*(robot.origin.xyz[1] - camera.position.y);
        camera.position.z += 0.1*(robot.origin.xyz[2] - camera.position.z);
    }
    else if (keyboard.pressed("x")) {
        camera.position.x -= 0.1*(robot.origin.xyz[0] - camera.position.x);
        camera.position.y -= 0.1*(robot.origin.xyz[1] - camera.position.y);
        camera.position.z -= 0.1*(robot.origin.xyz[2] - camera.position.z);
    }

    if (keyboard.pressed("w")) { 
        textbar.innerHTML = "moving base forward";
        robot.control.xyz[2] += 0.1 * (robot_heading[2][0] - robot.origin.xyz[2]);
        robot.control.xyz[0] += 0.1 * (robot_heading[0][0] - robot.origin.xyz[0]);
    }
    if (keyboard.pressed("s")) { 
        textbar.innerHTML = "moving base backward";
        robot.control.xyz[2] += -0.1 * (robot_heading[2][0] - robot.origin.xyz[2]);
        robot.control.xyz[0] += -0.1 * (robot_heading[0][0] - robot.origin.xyz[0]);
    }
    if (keyboard.pressed("a")) {
        textbar.innerHTML = "turning base left";
        robot.control.rpy[1] += 0.1;
    }
    if (keyboard.pressed("d")) {
        textbar.innerHTML = "turning base right";
        robot.control.rpy[1] += -0.1;
    }

    // other keys (ik control, motion planning, etc.) kept same as your original
}

//////////////////////////////////////////////////
/////     FAKE KEYBOARD PRESS BASED ON DEPTH VALUE
//////////////////////////////////////////////////

ws.onmessage = (e) => {
    let data = JSON.parse(e.data);
    console.log("Received from ZMQ via WebSocket:", data.forward);

    if (data.forward) {
      robot.control.xyz[2] += data.forward * (robot_heading[2][0] - robot.origin.xyz[2]);
      robot.control.xyz[0] += data.forward * (robot_heading[0][0] - robot.origin.xyz[0]);
    }
    if (data.yaw > 12) {
      robot.control.rpy[1] -= 0.1;
    }
    if (data.yaw < -12) {
      robot.control.rpy[1] += 0.1;
    }
      
  };




//////////////////////////////////////////////////
/////     OTHER FUNCTIONS (help display, pose setting)
//////////////////////////////////////////////////

kineval.displayHelp = function display_help () {
    textbar.innerHTML = "kineval user interface commands" 
        + "<br>mouse: rotate camera about robot base "
        + "<br>z/x : camera zoom with respect to base "
        + "<br>t : toggle starting point mode "
        + "<br>w/s a/d q/e : move base along forward/turning/strafe direction"
        + "<br>j/k/l : focus active joint to child/parent/sibling "
        + "<br>u/i : control active joint"
        + "<br>c : execute clock tick controller "
        + "<br>o : control robot arm to current setpoint target "
        + "<br>0 : control arm to zero pose "
        + "<br>Shift+[1-9] : assign current pose to a pose setpoint"
        + "<br>[1-9] : assign a pose setpoint to current setpoint target"
        + "<br>g : print pose setpoints to console "
        + "<br>p : iterate inverse kinematics motion "
        + "<br>r/f : move inverse kinematics target up/down"
        + "<br>m : invoke motion planner "
        + "<br>n/b : show next/previous pose in motion plan "
        + "<br>h : toggle gui command widget "
        + "<br>v : print commands to screen ";
}

kineval.toggleStartpointMode = function toggle_startpoint_mode() {
    textbar.innerHTML = "toggled startpoint mode";
    kineval.params.just_starting = !kineval.params.just_starting;
}

// -- and your other link changing functions (kineval.changeActiveLinkUp, Down, etc.) remain unchanged

