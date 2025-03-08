import React from 'react';
import { AppBar, Toolbar, Typography } from '@mui/material';
import styled from 'styled-components';

const NeonAppBar = styled(AppBar)`
  background: rgba(13, 13, 13, 0.8); /* Semi-transparent dark overlay */
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 15px rgba(0, 255, 204, 0.4);
`;

function Navbar() {
    return (
        <NeonAppBar position="static">
            <Toolbar>
                <Typography variant="h6">SheGuard</Typography>
            </Toolbar>
        </NeonAppBar>
    );
}

export default Navbar;