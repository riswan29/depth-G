$(document).ready(function(){

  $('#menu').click(function(){
    $(this).toggleClass('fa-times');
    $('header').toggleClass('toggle');
  });

  $(window).on('scroll load',function(){

    $('#menu').removeClass('fa-times');
    $('header').removeClass('toggle');

    if($(window).scrollTop() > 0){
      $('.top').show();
    }else{
      $('.top').hide();
    }

  });

  // smooth scrolling 

  $('a[href*="#"]').on('click',function(e){

    e.preventDefault();

    $('html, body').animate({

      scrollTop : $($(this).attr('href')).offset().top,

    },
      500, 
      'linear'
    );
  });
  $(document).on('click','.send', function(){
    /* Inputan Formulir */
    var input_name          = $("#name").val(),
        input_Email        = $("#Email").val(),   
        input_description   = $("#description").val();
 
    /* Pengaturan Whatsapp */
    var walink      = 'https://web.whatsapp.com/send',
        phone       = '62895414599871',
        text        = 'Haii Saya dari web Porto Anda !!!',
        text_yes    = 'Pesanan Anda berhasil terkirim.',
        text_no     = 'Isilah formulir terlebih dahulu.';
 
    /* Smartphone Support */
    if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
        var walink = 'whatsapp://send';
    }
 
    if(input_name != "" && input_Email != "" ){
        /* Whatsapp URL */
        var checkout_whatsapp = walink + '?phone=' + phone + '&text=' + text + '%0A%0A' +
            '*Nama* : ' + input_name + '%0A' +
            '*Email* : ' + input_Email+ '%0A' +
            '*Pesan* : ' + input_description + '%0A';
 
        /* Whatsapp Window Open */
        window.open(checkout_whatsapp,'_blank');
        document.getElementById("text-info").innerHTML = '<div class="alert alert-success">'+text_yes+'</div>';
    } else {
        document.getElementById("text-info").innerHTML = '<div class="alert alert-danger">'+text_no+'</div>';
    }
});
});