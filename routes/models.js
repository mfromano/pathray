var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/models', function(req, res, next) {
    res.render('../models', { title: 'Express' });
});

module.exports = router;