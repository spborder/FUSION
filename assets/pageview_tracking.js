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
    }
};