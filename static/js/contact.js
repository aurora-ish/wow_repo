const arrowLeft = document.querySelector(".arrow-left");
const arrowRight = document.querySelector(".arrow-right");
const dots = document.querySelectorAll(".dot");

const pages = ["/", "/ContactUs", "yourPurchases"];

const skipIndex = 2;

let currentPageIndex = 1;

function updateActiveDot() {
  dots.forEach((dot, index) => {
    dot.classList.remove("active");

    if (index === 1) {
      dot.classList.add("active");
    }
  });
}

updateActiveDot();

arrowLeft.addEventListener("click", () => {
  window.location.href = pages[0];
});

arrowRight.addEventListener("click", () => {
  window.location.href = pages[2];
});

dots.forEach((dot, index) => {
  dot.addEventListener("click", () => {
    if (index === 0) {
      window.location.href = "/";
    } else if (index === 1) {
      return;
    } else if (index === skipIndex) {
      return;
    } else if (index === 3) {
      window.location.href = "/yourPurchases";
    }
  });
});