// This one seems to work well ?? 
// I think i have noticed some problems with it, because
// this seems to be my original import_js script, that i later tried to find alternatives
// to, unsuccessfully. So that means that there must be something wrong with it,
// just not sure what exactly..
// UPDATE: the problem is that errors that are reported in firebug have incorrect place information.
// The other scripts, show correct script file and line number when an error occures, but they have other problems..
// This one works well, if we would overlook this flaw.

(function($)
{
    /*
     * $.import_js() helper (for javascript importing within javascript).
     */
    var import_js_imported = [];
    
    $.extend(true,
    {
        import_js : function(script)
        {
            var found = false;
            for (var i = 0; i < import_js_imported.length; i++)
                if (import_js_imported[i] == script) {
                    found = true;
                    break;
                }
            
            if (found == false) {
                $("head").append('<script type="text/javascript" src="' + script + '"></script>');
                import_js_imported.push(script);
            }
        }
    });
    
})(jQuery);