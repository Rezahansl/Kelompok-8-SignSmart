/* ============= Show Nav Menu ============= */
const navMenu = document.getElementById("nav-menu");
const navToggle = document.getElementById("nav-toggle");
const navClose = document.getElementById("nav-close");

/* Validate If for clicked nav toggle */
if (navToggle) {
  navToggle.addEventListener("click", () => {
    console.log("toggle");
    navMenu.classList.add("show-menu");
  });
}

/* Validate If for clicked nav close */
if (navClose) {
  navClose.addEventListener("click", () => {
    navMenu.classList.remove("show-menu");
  });
}

/* ============= Login ============= */
const login = document.getElementById("login"),
  loginBtn = document.getElementById("login-btn"),
  loginClose = document.getElementById("login-close");

loginBtn.addEventListener("click", () => {
  login.classList.add("show-login");
});

loginClose.addEventListener("click", () => {
  login.classList.remove("show-login");
});

/* ============= Remove Menu Mobile Screen ============= */
const navLink = document.querySelectorAll(".nav_link");

/* This function will remove the show menu class dari nav menu class */
function navLinkAction() {
  navMenu.classList.remove("show-menu");
}

/* Every time we click navLink, it will run the navLinkAction */
navLink.forEach((n) => n.addEventListener("click", navLinkAction));

/* ============= Change Background Color of Header ============= */
function scrollHeader() {
  const header = document.getElementById("header");

  if (this.scrollY >= 80) header.classList.add("scroll-header");
  else header.classList.remove("scroll-header");
}
window.addEventListener("scroll", scrollHeader);


/* ============= Scroll Up ============= */
function scrollUp() {
  const scrollUp = document.getElementById("scroll-up");
  if (this.scrollY >= 200) scrollUp.classList.add("show-scroll");
  else scrollUp.classList.remove("show-scroll");
}
window.addEventListener("scroll", scrollUp);

document
  .getElementById("result_container")
  .addEventListener("click", function () {
    document.getElementById("file_input").click();
  });

