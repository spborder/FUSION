if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.clientside = {
    trackPageView: function (pathname) {
        var eventInfo = {
            'event': 'Pageview',
            'pagePath': pathname
        };

        // Check if userId is available and add it to the event
        if (window.user_id) {
            eventInfo.user_id = window.user_id;
        }

        if (window.dataLayer) {
            window.dataLayer.push(eventInfo);
        }
        return null;
    },
    updateDataLayerWithUserId: function (userIdJson) {
        if (window.dataLayer && userIdJson) {
            var userIdInfo = JSON.parse(userIdJson);
            window.user_id = userIdInfo.user_id;
            window.dataLayer.push({
                'event': 'trigger config',
                'user_id': userIdInfo.user_id
            }) // Store userId globally for use in all events
        }
        return null;
    },
    trackManualRoiData: function (userAnn){
        if (window.dataLayer && userAnn) {
            var userAnnotationsInfo = JSON.parse(userAnn);
            window.dataLayer.push({
                'event': 'User Annotations',
                'user_ann_slide_name': userAnnotationsInfo.slide_name,
                'user_ann_item_id': userAnnotationsInfo.item_id,
                'geoJSON_info': userAnnotationsInfo.geoJSON_info
            })
        }
        return null;
    },
    trackPlugInData: function (pluginTrack){
        if (window.dataLayer && pluginTrack) {
            var pluginTrackInfo = JSON.parse(pluginTrack);
            window.dataLayer.push({
                'event': 'Plugin used',
                'plugin_used': pluginTrackInfo.plugin_used
            }) 
        }
        return null;
    },

};