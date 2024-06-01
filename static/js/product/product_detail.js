// JavaScript code

let currentSlide = 0;
const slides = document.querySelectorAll('.bbb_deals_image');
const totalSlides = slides.length;

function showSlide(n) {
    // Hide all slides
    for (let i = 0; i < totalSlides; i++) {
        slides[i].style.display = 'none';
    }
    // Display the slide with the given index
    slides[n].style.display = 'block';
}

function prevSlide() {
    currentSlide = (currentSlide - 1 + totalSlides) % totalSlides;
    showSlide(currentSlide);
}

function nextSlide() {
    currentSlide = (currentSlide + 1) % totalSlides;
    showSlide(currentSlide);
}

// Show the initial slide
showSlide(currentSlide);

// Resize images when the window is resized
window.addEventListener('resize', resizeImages);

// Resize images initially
resizeImages();



function get_full_title(element) {
    var rpTitleOverflow = element.nextElementSibling;
    rpTitleOverflow.style.display = 'block';
}

function hide_full_title(element) {
    setTimeout(() => {
        var rpTitleOverflow = element.nextElementSibling;
        rpTitleOverflow.style.display = 'none';
    }, 1000);

}



