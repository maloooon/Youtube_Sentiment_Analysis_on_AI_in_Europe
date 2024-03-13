


var chart = c3.generate({
    bindto: '#chart',
    data: {
      columns: [
        ['amount_comments', 250],
      ]

    },
    types: {
      amount_comments : 'bar'
    } ,

});

