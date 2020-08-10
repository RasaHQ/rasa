import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

export default (function () {
  if (!ExecutionEnvironment.canUseDOM) {
    return null;
  }

  return {
    onRouteUpdate({location}) {
      if (location) {
        setTimeout(() => {
          window.dataLayer.push({
            event: 'route-update',
          });
        }, 50);
      }
    },
  };
})();
